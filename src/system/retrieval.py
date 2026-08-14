import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .model_settings import ensure_hf_mirror, resolve_clip_model, resolve_colpali, resolve_embed_batch_size, resolve_embed_device

ensure_hf_mirror()

import faiss
import numpy as np

_ST_CLS = None


def _get_sentence_transformer_cls():
    """延迟导入：避免一启动就拉 nltk/pycrfsuite 整条重链；首次调用可能需数十秒。"""
    global _ST_CLS
    if _ST_CLS is not None:
        return _ST_CLS
    print(
        "正在加载 sentence_transformers（首次较慢，请耐心等待、勿按 Ctrl+C）…",
        flush=True,
    )
    try:
        from sentence_transformers import SentenceTransformer as _ST_CLS_REF
    except BaseException:
        import traceback

        traceback.print_exc()
        print(
            "导入 sentence_transformers 失败。"
            "可尝试：$env:FINANCE_EMBED_DEVICE='cpu' 再运行；或在 mineru 中执行 pip install -U sentence-transformers",
            flush=True,
        )
        raise
    _ST_CLS = _ST_CLS_REF
    print("sentence_transformers 已成功载入。", flush=True)
    return _ST_CLS


def _hash_file(path: Path) -> str:
    """对大文件分块计算 MD5，避免整文件读入内存。"""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _faiss_io_path(path: Path, *, for_write: bool = False) -> str:
    """
    Windows 下路径含中文或过长时，faiss 内部 fopen 易失败（误报 No such file or directory）。
    写入前确保父目录存在；必要时使用 \\?\\ 扩展路径。
    """
    p = Path(path).expanduser().resolve()
    if for_write:
        p.parent.mkdir(parents=True, exist_ok=True)
    s = os.fspath(p)
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        if any(ord(c) > 127 for c in s) or len(s) >= 240:
            s = "\\\\?\\" + s
    return s


def _faiss_write_index(index, path: Path) -> None:
    """
    写入 FAISS 索引。
    Windows 下 faiss 的 C++ fopen 常无法打开含中文的绝对路径（误报 No such file or directory），
    因此在扩展路径与普通绝对路径均失败后，改为 chdir 到目标目录再只传 ASCII 文件名。
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[BaseException] = None
    for cand in (_faiss_io_path(p, for_write=False), os.fspath(p)):
        try:
            faiss.write_index(index, cand)
            return
        except RuntimeError as e:
            last_err = e
    old = os.getcwd()
    try:
        os.chdir(p.parent)
        faiss.write_index(index, p.name)
        return
    except RuntimeError as e:
        last_err = e
    finally:
        try:
            os.chdir(old)
        except OSError:
            pass
    if last_err:
        raise last_err


def _faiss_read_index(path: Path):
    """读取 FAISS 索引；策略与 _faiss_write_index 一致（含中文路径时 chdir + 相对文件名）。"""
    p = Path(path).expanduser().resolve()
    last_err: Optional[BaseException] = None
    for cand in (_faiss_io_path(p, for_write=False), os.fspath(p)):
        try:
            return faiss.read_index(cand)
        except RuntimeError as e:
            last_err = e
    old = os.getcwd()
    try:
        os.chdir(p.parent)
        return faiss.read_index(p.name)
    except RuntimeError as e:
        last_err = e
    finally:
        try:
            os.chdir(old)
        except OSError:
            pass
    if last_err:
        raise last_err


class KnowledgeStore:
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.blocks: List[Dict] = []
        self.page_images: Dict[Tuple[str, int], str] = {}

    @staticmethod
    def _ingest_pages_from_doc(
        doc: Dict,
        blocks: List[Dict],
        page_images: Dict[Tuple[str, int], str],
    ) -> None:
        doc_name = doc.get("doc_id", doc.get("doc_name", "unknown_doc"))
        for page in doc.get("pages", []) or []:
            page_no = int(page.get("page_no", 1))
            page_meta = page.get("meta", {}) or {}
            if page_meta.get("image_ref"):
                page_images[(doc_name, page_no)] = page_meta["image_ref"]

            for block in page.get("blocks", []) or []:
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                block_meta = block.get("meta", {}) or {}
                image_ref = block_meta.get("image_ref") or page_meta.get("image_ref")
                if image_ref:
                    page_images[(doc_name, page_no)] = image_ref
                blocks.append(
                    {
                        "source_doc": doc_name,
                        "page": page_no,
                        "block_id": block.get("block_id", f"{doc_name}_p{page_no}"),
                        "type": block.get("block_type", "paragraph"),
                        "bbox": block.get("bbox"),
                        "content": text,
                        "image_ref": image_ref,
                        "score": 0.0,
                    }
                )

    @staticmethod
    def _ingest_flat_doc(
        doc: Dict,
        blocks: List[Dict],
        page_images: Dict[Tuple[str, int], str],
    ) -> None:
        doc_name = doc.get("doc_id", doc.get("doc_name", "unknown_doc"))
        raw = (doc.get("content") or doc.get("text") or "").strip()
        if not raw:
            return
        page_no = int(doc.get("page_no", 1) or 1)
        meta = doc.get("metadata", {}) or {}
        image_ref = meta.get("image_ref")
        if image_ref:
            page_images[(doc_name, page_no)] = image_ref
        blocks.append(
            {
                "source_doc": doc_name,
                "page": page_no,
                "block_id": doc.get("block_id") or f"{doc_name}_p{page_no}_full",
                "type": "document",
                "bbox": None,
                "content": raw,
                "image_ref": image_ref,
                "score": 0.0,
            }
        )

    @staticmethod
    def _doc_matches(doc: Dict, keywords: Optional[List[str]]) -> bool:
        if not keywords:
            return True
        parts = [
            str(doc.get("doc_id") or ""),
            str(doc.get("doc_name") or ""),
            str(doc.get("title") or ""),
            str((doc.get("metadata") or {}).get("source") or ""),
            str(doc.get("content") or "")[:2000],
            str(doc.get("text") or "")[:2000],
        ]
        # 文档 id 常为哈希，需扫前几页正文
        for page in (doc.get("pages") or [])[:3]:
            for block in (page.get("blocks") or [])[:8]:
                parts.append(str(block.get("text") or "")[:500])
        blob = "\n".join(parts)
        return any(k in blob for k in keywords)

    def load(
        self,
        max_blocks: Optional[int] = None,
        doc_keywords: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        加载知识块。
        max_blocks / doc_keywords 用于低内存演示：只保留标题匹配的文档，并截断块数。
        """
        blocks: List[Dict] = []
        page_images: Dict[Tuple[str, int], str] = {}
        path = self.kb_path
        large_kb = path.is_file() and path.stat().st_size > 80 * 1024 * 1024
        keywords = [k for k in (doc_keywords or []) if k]

        def _ingest_doc(doc: Dict, *, flat: bool = False) -> bool:
            """返回 True 表示已达 max_blocks，应停止。"""
            if not self._doc_matches(doc, keywords or None):
                return False
            before = len(blocks)
            if flat:
                self._ingest_flat_doc(doc, blocks, page_images)
            else:
                self._ingest_pages_from_doc(doc, blocks, page_images)
            if max_blocks is not None and len(blocks) > max_blocks:
                del blocks[max_blocks:]
                return True
            # 无新增块时不提前停
            _ = before
            return max_blocks is not None and len(blocks) >= max_blocks

        try:
            import ijson  # noqa: WPS433 — 超大 knowledge_base.json 流式解析
        except ImportError as e:
            if large_kb:
                raise ImportError(
                    f"知识库文件较大（约 {path.stat().st_size // (1024 * 1024)} MB），"
                    "必须安装 ijson 才能流式加载，否则会 MemoryError。"
                    "请在当前环境执行：pip install ijson"
                ) from e
            data = json.loads(path.read_text(encoding="utf-8"))
            for doc in data.get("documents", []):
                if _ingest_doc(doc, flat=False):
                    break
            if not blocks:
                for doc in data.get("documents", []):
                    if _ingest_doc(doc, flat=True):
                        break
        else:
            try:
                with path.open("rb") as f:
                    for doc in ijson.items(f, "documents.item"):
                        if _ingest_doc(doc, flat=False):
                            break
                if not blocks:
                    with path.open("rb") as f:
                        for doc in ijson.items(f, "documents.item"):
                            if _ingest_doc(doc, flat=True):
                                break
            except MemoryError as e:
                raise MemoryError(
                    "加载 knowledge_base.json 时内存仍不足。"
                    "可尝试关闭其它程序、增加虚拟内存，或设置 FINANCE_MAX_BLOCKS / FINANCE_DEMO_KEYWORDS 精简加载。"
                ) from e

        self.blocks = blocks
        self.page_images = page_images
        return blocks


class TextRetriever:
    def __init__(self, model_name: str, cache_dir: str, kb_hash: str, device: Optional[str] = None):
        self.device = device if device is not None else resolve_embed_device()
        self.embed_batch_size = resolve_embed_batch_size(self.device)
        # Windows 上多线程 + 大模型权重易原生闪退
        try:
            import torch

            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1") or "1")))
        except Exception:
            pass
        ST = _get_sentence_transformer_cls()
        print(f"正在加载嵌入模型: {model_name} (device={self.device}) …", flush=True)
        self.model = ST(model_name, device=self.device)
        print("嵌入模型加载完成。", flush=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_hash = kb_hash
        self.index = None
        self.metadata: List[Dict] = []

    def _cache_paths(self) -> Tuple[Path, Path]:
        dim = self.model.get_sentence_embedding_dimension()
        prefix = f"{self.kb_hash}_{dim}_text"
        return self.cache_dir / f"{prefix}.faiss", self.cache_dir / f"{prefix}.pkl"

    def build(self, blocks: List[Dict], force_rebuild: bool = False) -> None:
        index_path, meta_path = self._cache_paths()
        if not force_rebuild and index_path.exists() and meta_path.exists():
            self.index = _faiss_read_index(index_path)
            self.metadata = pickle.loads(meta_path.read_bytes())
            return

        texts = [b["content"] for b in blocks]
        if not texts:
            raise ValueError(
                "知识库中没有任何非空文本块，无法构建文本向量索引。"
                "请检查 knowledge_base.json：需包含 documents → pages → blocks（block.text），"
                "或 documents[].content / documents[].text 全文字段。"
            )

        # 分批 encode + 写入 FAISS，避免一次性分配 (N, D) 大数组导致 MemoryError
        dim = int(self.model.get_sentence_embedding_dimension())
        self.index = faiss.IndexFlatIP(dim)
        bs = max(1, int(self.embed_batch_size))
        n = len(texts)
        print(f"文本向量分批构建：{n} 块，batch={bs}，dim={dim}", flush=True)
        for start in range(0, n, bs):
            batch = texts[start : start + bs]
            emb = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if emb.shape[0] != len(batch) or emb.shape[1] != dim:
                raise ValueError(
                    f"嵌入维度异常: batch={len(batch)} dim={dim}，实际 shape={emb.shape}。"
                )
            faiss.normalize_L2(emb)
            self.index.add(emb)
            done = min(start + bs, n)
            if done == n or done % max(bs * 20, 1) < bs:
                print(f"  文本向量进度 {done}/{n}", flush=True)

        self.metadata = blocks
        _faiss_write_index(self.index, index_path)
        meta_path.write_bytes(pickle.dumps(self.metadata))
        print(f"文本向量已缓存: {index_path.name}", flush=True)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.index is None:
            return []
        q = self.model.encode([query], batch_size=1, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["score"] = float(score)
                item["retrieval_channel"] = "text"
                out.append(item)
        return out


class VisionPageRetriever:
    """
    真页图向量检索：
    - backend=colpali：优先使用 ColPali 模型编码页面图像
    - backend=visrag：使用 CLIP 风格视觉编码（VisRAG风格）
    两者都失败时回退 proxy 检索，保证链路可运行。
    """

    def __init__(
        self,
        cache_dir: str,
        kb_hash: str,
        backend: str = "visrag",
        model_name: Optional[str] = None,
        text_proxy_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_hash = kb_hash
        self.backend = backend
        self.device = device if device is not None else resolve_embed_device()
        self.embed_batch_size = resolve_embed_batch_size(self.device)
        self.model_name = model_name or (
            resolve_colpali() if backend == "colpali" else resolve_clip_model()
        )
        self.index = None
        self.metadata: List[Dict] = []
        proxy_id = text_proxy_model or "BAAI/bge-m3"
        ST = _get_sentence_transformer_cls()
        self._text_proxy = ST(proxy_id, device=self.device)
        self._colpali_bundle = None
        self._visrag_bundle = None

    def _cache_paths(self) -> Tuple[Path, Path]:
        prefix = f"{self.kb_hash}_{self.backend}_image"
        return self.cache_dir / f"{prefix}.faiss", self.cache_dir / f"{prefix}.pkl"

    def build(self, blocks: List[Dict], page_images: Dict[Tuple[str, int], str], force_rebuild: bool = False) -> None:
        index_path, meta_path = self._cache_paths()
        if not force_rebuild and index_path.exists() and meta_path.exists():
            self.index = _faiss_read_index(index_path)
            self.metadata = pickle.loads(meta_path.read_bytes())
            return

        page_items = self._build_page_items(blocks, page_images)
        if not page_items:
            return

        embeddings = self._encode_page_items(page_items)
        if embeddings is None:
            # fallback 到文本代理页向量，保证可运行
            embeddings = self._text_proxy.encode(
                [x["content"] for x in page_items],
                batch_size=self.embed_batch_size,
                convert_to_numpy=True,
            ).astype(np.float32)
            for item in page_items:
                item["retrieval_channel"] = "image_proxy"
        else:
            for item in page_items:
                item["retrieval_channel"] = f"image_{self.backend}"

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = page_items
        _faiss_write_index(self.index, index_path)
        meta_path.write_bytes(pickle.dumps(self.metadata))

    @staticmethod
    def _build_page_items(blocks: List[Dict], page_images: Dict[Tuple[str, int], str]) -> List[Dict]:
        grouped: Dict[Tuple[str, int], List[Dict]] = {}
        for b in blocks:
            key = (b["source_doc"], int(b["page"]))
            grouped.setdefault(key, []).append(b)

        items: List[Dict] = []
        for (doc, page), rows in grouped.items():
            image_ref = page_images.get((doc, page)) or rows[0].get("image_ref")
            items.append(
                {
                    "source_doc": doc,
                    "page": page,
                    "block_id": f"{doc}_p{page}_img",
                    "type": "page_image",
                    "image_ref": image_ref,
                    "content": "\n".join([r["content"] for r in rows])[:6000],
                    "score": 0.0,
                }
            )
        return items

    def _encode_page_items(self, page_items: List[Dict]) -> Optional[np.ndarray]:
        if self.backend == "colpali":
            return self._encode_with_colpali(page_items)
        return self._encode_with_visrag(page_items)

    def _encode_with_colpali(self, page_items: List[Dict]) -> Optional[np.ndarray]:
        bundle = self._load_colpali()
        if bundle is None:
            return None
        import torch
        processor, model, device = bundle
        from PIL import Image

        vectors = []
        for item in page_items:
            image_ref = item.get("image_ref")
            if not image_ref or not Path(image_ref).exists():
                vectors.append(None)
                continue
            try:
                with Image.open(image_ref) as img:
                    img = img.convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                vec = self._pick_output_vector(outputs)
                vectors.append(vec)
            except Exception:
                vectors.append(None)

        if all(v is None for v in vectors):
            return None

        dim = next(v.shape[0] for v in vectors if v is not None)
        final = []
        for idx, v in enumerate(vectors):
            if v is None:
                proxy = self._text_proxy.encode(
                    [page_items[idx]["content"]], batch_size=1, convert_to_numpy=True
                ).astype(np.float32)[0]
                if proxy.shape[0] != dim:
                    proxy = proxy[:dim] if proxy.shape[0] > dim else np.pad(proxy, (0, dim - proxy.shape[0]))
                final.append(proxy)
            else:
                final.append(v)
        return np.array(final, dtype=np.float32)

    def _encode_with_visrag(self, page_items: List[Dict]) -> Optional[np.ndarray]:
        bundle = self._load_visrag()
        if bundle is None:
            return None
        import torch
        processor, model, device = bundle
        from PIL import Image

        vectors = []
        for item in page_items:
            image_ref = item.get("image_ref")
            if not image_ref or not Path(image_ref).exists():
                vectors.append(None)
                continue
            try:
                with Image.open(image_ref) as img:
                    img = img.convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    vec = model.get_image_features(**inputs).detach().cpu().numpy()[0]
                vectors.append(vec.astype(np.float32))
            except Exception:
                vectors.append(None)

        if all(v is None for v in vectors):
            return None
        dim = next(v.shape[0] for v in vectors if v is not None)
        final = []
        for idx, v in enumerate(vectors):
            if v is None:
                proxy = self._text_proxy.encode(
                    [page_items[idx]["content"]], batch_size=1, convert_to_numpy=True
                ).astype(np.float32)[0]
                if proxy.shape[0] != dim:
                    proxy = proxy[:dim] if proxy.shape[0] > dim else np.pad(proxy, (0, dim - proxy.shape[0]))
                final.append(proxy)
            else:
                final.append(v)
        return np.array(final, dtype=np.float32)

    @staticmethod
    def _pick_output_vector(outputs) -> np.ndarray:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.detach().cpu().numpy()[0].astype(np.float32)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            tensor = outputs.last_hidden_state.mean(dim=1)
            return tensor.detach().cpu().numpy()[0].astype(np.float32)
        if isinstance(outputs, tuple) and len(outputs) > 0:
            tensor = outputs[0]
            if len(tensor.shape) == 3:
                tensor = tensor.mean(dim=1)
            return tensor.detach().cpu().numpy()[0].astype(np.float32)
        raise RuntimeError("Unsupported model outputs for image embedding")

    def _encode_query(self, query: str) -> np.ndarray:
        if self.backend == "colpali":
            vec = self._encode_query_colpali(query)
            if vec is not None:
                return vec
        else:
            vec = self._encode_query_visrag(query)
            if vec is not None:
                return vec
        return self._text_proxy.encode([query], batch_size=1, convert_to_numpy=True).astype(np.float32)[0]

    def _encode_query_colpali(self, query: str) -> Optional[np.ndarray]:
        bundle = self._load_colpali()
        if bundle is None:
            return None
        processor, model, device = bundle
        try:
            import torch
            inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            return self._pick_output_vector(outputs)
        except Exception:
            return None

    def _encode_query_visrag(self, query: str) -> Optional[np.ndarray]:
        bundle = self._load_visrag()
        if bundle is None:
            return None
        processor, model, device = bundle
        try:
            import torch
            inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                vec = model.get_text_features(**inputs).detach().cpu().numpy()[0]
            return vec.astype(np.float32)
        except Exception:
            return None

    def _load_colpali(self):
        if self._colpali_bundle is not None:
            return self._colpali_bundle
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            model.eval()
            device = resolve_embed_device()
            model.to(device)
            self._colpali_bundle = (processor, model, device)
            return self._colpali_bundle
        except Exception:
            return None

    def _load_visrag(self):
        if self._visrag_bundle is not None:
            return self._visrag_bundle
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model.eval()
            device = resolve_embed_device()
            model.to(device)
            self._visrag_bundle = (processor, model, device)
            return self._visrag_bundle
        except Exception:
            return None

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        if self.index is None:
            return []
        q = self._encode_query(query).astype(np.float32)
        q = np.expand_dims(q, axis=0)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["score"] = float(score)
                out.append(item)
        return out


class FusionRetriever:
    def __init__(self, text_retriever: TextRetriever, image_retriever: VisionPageRetriever):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever

    @staticmethod
    def _rrf_rank(results: List[Dict], k: int = 60) -> Dict[str, float]:
        rank_scores: Dict[str, float] = {}
        for rank, item in enumerate(results, start=1):
            key = item["block_id"]
            rank_scores[key] = rank_scores.get(key, 0.0) + 1.0 / (k + rank)
        return rank_scores

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        text_results = self.text_retriever.search(query, top_k=max(12, top_k))
        image_results = self.image_retriever.search(query, top_k=max(8, top_k // 2))
        text_rrf = self._rrf_rank(text_results)
        image_rrf = self._rrf_rank(image_results)

        merged: Dict[str, Dict] = {}
        for item in text_results + image_results:
            bid = item["block_id"]
            if bid not in merged:
                merged[bid] = dict(item)
            merged[bid]["fusion_score"] = text_rrf.get(bid, 0.0) + image_rrf.get(bid, 0.0)
            merged[bid]["text_rrf"] = text_rrf.get(bid, 0.0)
            merged[bid]["image_rrf"] = image_rrf.get(bid, 0.0)

        ranked = sorted(merged.values(), key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        return ranked[:top_k]


class Neo4jGraphRetriever:
    """
    可选图检索通道（阶段3）：
    - 优先查询 Neo4j（若连接可用）
    - 若不可用，回退读取本地 kg_triples.json 做轻量匹配
    """

    def __init__(
        self,
        kg_dir: str = "./output",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self.kg_dir = Path(kg_dir)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.local_triples: List[Dict] = []
        self._init_backend()

    def _init_backend(self):
        if self.neo4j_uri and self.neo4j_user and self.neo4j_password:
            try:
                from neo4j import GraphDatabase

                self.driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
                return
            except Exception:
                self.driver = None
        triples_path = self.kg_dir / "kg_triples.json"
        if triples_path.exists():
            self.local_triples = json.loads(triples_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        if self.driver is not None:
            out = self._search_neo4j(query, top_k=top_k)
            if out:
                return out
        return self._search_local(query, top_k=top_k)

    def _search_neo4j(self, query: str, top_k: int = 8) -> List[Dict]:
        cql = """
        MATCH (h)-[r]->(t)
        WHERE h.name CONTAINS $q OR t.name CONTAINS $q
        RETURN h.name AS head, type(r) AS relation, t.name AS tail, r.evidence AS evidence, r.source_doc AS source_doc, r.page AS page, r.block_id AS block_id
        LIMIT $k
        """
        try:
            with self.driver.session() as session:
                rows = session.run(cql, q=query, k=top_k).data()
            return [
                {
                    "block_id": row.get("block_id") or f"kg_{idx}",
                    "source_doc": row.get("source_doc"),
                    "page": row.get("page"),
                    "type": "kg_triple",
                    "content": f"{row.get('head')} -[{row.get('relation')}]-> {row.get('tail')}",
                    "score": 0.75 - idx * 0.01,
                    "fusion_score": 0.75 - idx * 0.01,
                    "retrieval_channel": "graph_neo4j",
                    "kg_evidence": row.get("evidence"),
                }
                for idx, row in enumerate(rows)
            ]
        except Exception:
            return []

    def _search_local(self, query: str, top_k: int = 8) -> List[Dict]:
        hits = []
        for tri in self.local_triples:
            blob = " ".join(
                [
                    str(tri.get("head", "")),
                    str(tri.get("relation", "")),
                    str(tri.get("tail", "")),
                    str(tri.get("evidence", "")),
                ]
            )
            if query in blob:
                hits.append(tri)
        out = []
        for idx, tri in enumerate(hits[:top_k]):
            out.append(
                {
                    "block_id": tri.get("block_id", f"kg_local_{idx}"),
                    "source_doc": tri.get("source_doc"),
                    "page": tri.get("page"),
                    "type": "kg_triple",
                    "content": f"{tri.get('head')} -[{tri.get('relation')}]-> {tri.get('tail')}",
                    "score": 0.7 - idx * 0.01,
                    "fusion_score": 0.7 - idx * 0.01,
                    "retrieval_channel": "graph_local",
                    "kg_evidence": tri.get("evidence"),
                }
            )
        return out


class HybridRetriever:
    def __init__(self, fusion_retriever: FusionRetriever, graph_retriever: Optional[Neo4jGraphRetriever] = None):
        self.fusion_retriever = fusion_retriever
        self.graph_retriever = graph_retriever

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        dense = self.fusion_retriever.search(query, top_k=top_k)
        if self.graph_retriever is None:
            return dense
        graph = self.graph_retriever.search(query, top_k=max(4, top_k // 2))
        merged = {x["block_id"]: x for x in dense}
        for g in graph:
            if g["block_id"] not in merged:
                merged[g["block_id"]] = g
        ranked = sorted(merged.values(), key=lambda x: x.get("fusion_score", x.get("score", 0.0)), reverse=True)
        return ranked[:top_k]


class _EmptyImageRetriever:
    """演示/低内存模式：跳过图像塔，避免再加载一套向量模型。"""

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        _ = query, top_k
        return []


def _env_max_blocks() -> Optional[int]:
    raw = (os.getenv("FINANCE_MAX_BLOCKS") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def _env_demo_keywords() -> Optional[List[str]]:
    raw = (os.getenv("FINANCE_DEMO_KEYWORDS") or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace("，", "|").split("|") if p.strip()]
    return parts or None


def _env_skip_image() -> bool:
    return (os.getenv("FINANCE_SKIP_IMAGE") or "").strip().lower() in {"1", "true", "yes", "y"}


def build_retrieval_stack(
    kb_path: str,
    cache_dir: str,
    text_model: str,
    image_proxy_model: str,
    image_backend: str = "visrag",
    enable_graph: bool = False,
    kg_dir: str = "./output",
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    force_rebuild: bool = False,
    max_blocks: Optional[int] = None,
    doc_keywords: Optional[List[str]] = None,
    skip_image: Optional[bool] = None,
):
    _ = image_proxy_model
    kb_file = Path(kb_path)
    max_blocks = max_blocks if max_blocks is not None else _env_max_blocks()
    doc_keywords = doc_keywords if doc_keywords is not None else _env_demo_keywords()
    skip_image = _env_skip_image() if skip_image is None else skip_image

    base_hash = _hash_file(kb_file)
    subset_tag = ""
    if max_blocks or doc_keywords:
        kw_key = ",".join(doc_keywords or [])
        subset_tag = hashlib.md5(f"{max_blocks}|{kw_key}".encode("utf-8")).hexdigest()[:10]
    kb_hash = f"{base_hash}_demo{subset_tag}" if subset_tag else base_hash

    store = KnowledgeStore(kb_path)
    print(
        f"加载知识库… max_blocks={max_blocks} keywords={doc_keywords} skip_image={skip_image}",
        flush=True,
    )
    blocks = store.load(max_blocks=max_blocks, doc_keywords=doc_keywords)
    if not blocks and doc_keywords:
        print("关键词未命中文档，回退为仅按 FINANCE_MAX_BLOCKS 截断加载…", flush=True)
        subset_tag = hashlib.md5(f"{max_blocks}|FALLBACK".encode("utf-8")).hexdigest()[:10]
        kb_hash = f"{base_hash}_demo{subset_tag}"
        blocks = store.load(max_blocks=max_blocks or 8000, doc_keywords=None)
    print(f"已加载文本块: {len(blocks)}", flush=True)
    if not blocks:
        raise ValueError(
            "精简加载后知识块为 0。请放宽 FINANCE_DEMO_KEYWORDS 或增大 FINANCE_MAX_BLOCKS。"
        )

    embed_device = resolve_embed_device()
    text = TextRetriever(model_name=text_model, cache_dir=cache_dir, kb_hash=kb_hash, device=embed_device)
    text.build(blocks, force_rebuild=force_rebuild)

    if skip_image:
        image = _EmptyImageRetriever()
        print("已跳过图像检索塔（FINANCE_SKIP_IMAGE）。", flush=True)
    else:
        image_model = resolve_colpali() if image_backend == "colpali" else resolve_clip_model()
        image = VisionPageRetriever(
            cache_dir=cache_dir,
            kb_hash=kb_hash,
            backend=image_backend,
            model_name=image_model,
            text_proxy_model=text_model,
            device=embed_device,
        )
        image.build(blocks=blocks, page_images=store.page_images, force_rebuild=force_rebuild)

    fusion = FusionRetriever(text_retriever=text, image_retriever=image)
    graph_retriever = None
    if enable_graph:
        graph_retriever = Neo4jGraphRetriever(
            kg_dir=kg_dir,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
    hybrid = HybridRetriever(fusion_retriever=fusion, graph_retriever=graph_retriever)
    return store, text, image, hybrid
