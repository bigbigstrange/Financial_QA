import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_settings import ensure_hf_mirror, resolve_clip_model, resolve_colpali, resolve_embed_batch_size, resolve_embed_device
from src.pipeline.docir import DocIR
from src.pipeline.snapshots import SnapshotManifest, read_manifest
from .sparse import SparseRetriever
from .structured import StructuredRetriever
from .vision import page_image_key
from .index_fingerprint import (
    IndexFingerprint,
    index_meta_matches,
    load_index_fingerprint_meta,
    write_index_fingerprint_meta,
)

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

    # ---- 检索单元（chunk）加载：KB v3 主路径 ---------------------------------
    #
    # 阶段 A 的切块器把行级碎片聚合成 400+ 字的语义单元，若检索层仍按 block 加载，
    # 切块成果就完全用不上。因此 KB v3 的 `chunks.jsonl` 优先级最高。

    @staticmethod
    def _resolve_chunks_path(kb_path: Path) -> Optional[Path]:
        """定位 chunks.jsonl：可能是路径本身、同目录兄弟文件，或目录内文件。"""
        if kb_path.is_file():
            if kb_path.name == "chunks.jsonl":
                return kb_path
            sibling = kb_path.with_name("chunks.jsonl")
            return sibling if sibling.is_file() else None
        if kb_path.is_dir():
            candidate = kb_path / "chunks.jsonl"
            return candidate if candidate.is_file() else None
        return None

    @staticmethod
    def _ingest_chunk(item: Dict, blocks: List[Dict]) -> None:
        """把 chunks.jsonl 的一行规范成检索层 runtime block。

        `Chunk.to_retrieval_item()` 已经产出了与检索层一致的字段命名，
        这里只做兜底补齐，保证下游（FAISS/BM25/结构化/图通道）无需感知来源差异。
        """
        content = (item.get("content") or "").strip()
        if not content:
            return
        chunk_id = item.get("chunk_id") or item.get("block_id")
        if not chunk_id:
            return
        page_start = item.get("page_start", item.get("page"))
        page_end = item.get("page_end", page_start)
        blocks.append(
            {
                "block_id": chunk_id,
                "chunk_id": chunk_id,
                "block_ids": list(item.get("block_ids") or []),
                "content": content,
                "source_doc": item.get("source_doc") or "unknown_doc",
                "document_id": item.get("document_id"),
                "document_version": item.get("document_version"),
                "snapshot_id": item.get("snapshot_id"),
                "page": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "bbox": item.get("bbox"),
                "section_path": item.get("section_path"),
                "type": item.get("type") or "paragraph",
                "parent_block_id": item.get("parent_block_id"),
                "char_len": item.get("char_len") or len(content),
                "token_estimate": item.get("token_estimate") or 0,
                "overlap_prev_chars": item.get("overlap_prev_chars") or 0,
                "trailing_prose_end": item.get("trailing_prose_end"),
                "quality_flags": list(item.get("quality_flags") or []),
                "source_format": item.get("source_format"),
                "image_ref": item.get("image_ref"),
                "score": 0.0,
            }
        )

    @staticmethod
    def _chunk_matches(item: Dict, keywords: Optional[List[str]]) -> bool:
        if not keywords:
            return True
        blob = "\n".join(
            [
                str(item.get("source_doc") or ""),
                str(item.get("section_path") or ""),
                str(item.get("document_id") or ""),
                str(item.get("content") or "")[:2000],
            ]
        )
        return any(k in blob for k in keywords)

    def _load_chunks(
        self,
        path: Path,
        *,
        max_blocks: Optional[int],
        doc_keywords: Optional[List[str]],
    ) -> List[Dict]:
        blocks: List[Dict] = []
        keywords = [k for k in (doc_keywords or []) if k] or None
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not self._chunk_matches(item, keywords):
                    continue
                self._ingest_chunk(item, blocks)
                if max_blocks is not None and len(blocks) >= max_blocks:
                    break
        return blocks

    def _load_page_images_from_manifest(self) -> Dict[Tuple[str, int], str]:
        """从快照 manifest 收集页图引用（视觉检索通道依赖它）。

        旧链路的页图路径全部指向已失效的外机绝对路径，导致图像通道 100% 失效；
        KB v3 在解析阶段就地渲染页图，这里把相对路径收上来供检索层解析。
        """
        images: Dict[Tuple[str, int], str] = {}
        manifest_path = self.kb_path.with_name("kb_v3_manifest.json")
        if not manifest_path.is_file():
            manifest_path = self.kb_path
        try:
            manifest = read_manifest(manifest_path)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError):
            return images
        base_dir = manifest_path.parent
        for artifact in manifest.versions:
            docir_path = Path(artifact.docir_path.replace('\\', '/'))
            if not docir_path.is_absolute():
                docir_path = base_dir / docir_path
            if not docir_path.is_file():
                continue
            try:
                doc_ir = DocIR.from_dict(
                    json.loads(docir_path.read_text(encoding="utf-8"))
                )
            except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError):
                continue
            doc_key = (doc_ir.document_version if doc_ir.document_version != "legacy-v1"
                       else doc_ir.document_id) or doc_ir.document_id
            for page in doc_ir.pages:
                ref = (page.meta or {}).get("image_ref")
                if not ref:
                    continue
                resolved = Path(ref.replace('\\', '/'))
                if not resolved.is_absolute():
                    # 引用统一为「相对 KB 根目录」（见 kb_builder._rebase_refs）
                    resolved = base_dir / resolved
                if resolved.is_file():
                    images[(doc_key, page.page_no)] = str(resolved)
        return images

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
                        # 以下字段旧版被整段丢弃，导致结构化通道的章节匹配、
                        # 表格证据、机构/日期过滤在主链路上全部失效。
                        "section_path": block.get("section_path"),
                        "parent_id": block.get("parent_id"),
                        "structural_locator": block.get("structural_locator"),
                        "caption": block.get("caption"),
                        "unit": block.get("unit"),
                        "table_headers": block.get("table_headers") or [],
                        "table_rows": block.get("table_rows") or [],
                        "table_md": block.get("table_md"),
                        "formula_latex": block.get("formula_latex"),
                        "ocr_used": bool(block.get("ocr_used")),
                        "quality_flags": list(block.get("quality_flags") or []),
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
        if not path.exists():
            raise FileNotFoundError(f"知识库路径不存在: {path}")

        # 优先级 1：KB v3 的 chunks.jsonl（阶段 A 的切块成果，检索单元质量最高）
        chunks_path = self._resolve_chunks_path(path)
        if chunks_path is not None:
            loaded = self._load_chunks(
                chunks_path, max_blocks=max_blocks, doc_keywords=doc_keywords
            )
            if loaded:
                self.blocks = loaded
                self.page_images = self._load_page_images_from_manifest()
                print(
                    f"已从 chunks.jsonl 加载检索单元: {len(loaded)}"
                    f"（页图 {len(self.page_images)} 张）",
                    flush=True,
                )
                return self.blocks

        if path.is_file():
            try:
                manifest = read_manifest(path)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                manifest = None
            if manifest is not None:
                self.blocks, self.page_images = self._load_manifest(
                    manifest,
                    max_blocks=max_blocks,
                    doc_keywords=doc_keywords,
                )
                return self.blocks
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

    def _load_manifest(
        self,
        manifest: SnapshotManifest,
        *,
        max_blocks: Optional[int],
        doc_keywords: Optional[List[str]],
    ) -> Tuple[List[Dict], Dict[Tuple[str, int], str]]:
        blocks: List[Dict] = []
        page_images: Dict[Tuple[str, int], str] = {}
        for artifact in manifest.versions:
            docir_path = Path(artifact.docir_path)
            if not docir_path.is_absolute():
                docir_path = self.kb_path.parent / docir_path
            if not docir_path.is_file():
                raise FileNotFoundError(f"快照引用缺失 DocIR: {docir_path}")
            doc_ir = DocIR.from_dict(json.loads(docir_path.read_text(encoding="utf-8")))
            for page in doc_ir.pages:
                page_no = page.page_no
                if page.meta.get("image_ref"):
                    page_images[(doc_ir.document_id, page_no)] = page.meta["image_ref"]
                for block in page.blocks:
                    text = (block.text or "").strip()
                    if not text:
                        continue
                    item = {
                        "snapshot_id": manifest.snapshot_id,
                        "document_id": doc_ir.document_id,
                        "document_version": doc_ir.document_version,
                        "source_doc": doc_ir.source_name or doc_ir.doc_id,
                        "page": page_no,
                        "block_id": block.block_id,
                        "type": block.block_type,
                        "bbox": block.bbox,
                        "content": text,
                        "section_path": block.section_path,
                        "structural_locator": block.structural_locator,
                        "parent_id": block.parent_id,
                        "quality_flags": block.quality_flags,
                        "image_ref": block.meta.get("image_ref")
                        or page.meta.get("image_ref"),
                        "score": 0.0,
                    }
                    if block.meta.get("image_ref") or page.meta.get("image_ref"):
                        page_images[(doc_ir.document_id, page_no)] = (
                            block.meta.get("image_ref")
                            or page.meta.get("image_ref")
                        )
                    blocks.append(item)
                    if max_blocks is not None and len(blocks) >= max_blocks:
                        return blocks, page_images
        return blocks, page_images


class TextRetriever:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        kb_hash: str,
        device: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        snapshot_config: Optional[Dict] = None,
    ):
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
        self.model_name = model_name
        # 限制最大序列长度：BGE-M3 默认 8192，而 torch 2.1.0 下 transformers 不启用 SDPA，
        # 退回 eager attention 后显存随序列长度增长（实测编码 2,910 块峰值约 8.2GB）。
        # 本项目 chunk 长度中位数约 390 字，512 token 足以覆盖绝大多数内容。
        max_seq = (os.environ.get("FINANCE_EMBED_MAX_SEQ_LEN") or "").strip()
        max_seq_len = int(max_seq) if max_seq.isdigit() and int(max_seq) > 0 else 512
        try:
            self.model.max_seq_length = max_seq_len
            print(f"嵌入最大序列长度: {max_seq_len}", flush=True)
        except Exception:  # noqa: BLE001 — 模型不支持该属性时忽略
            pass
        print("嵌入模型加载完成。", flush=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_hash = kb_hash
        self.snapshot_id = snapshot_id
        self.snapshot_config = snapshot_config or {}
        try:
            revision = self.model[0].auto_model.config._commit_hash or "local-unversioned"
        except (AttributeError, TypeError, KeyError, IndexError):
            revision = "local-unversioned"
        self.embedding_revision = f"{revision}:max_seq={max_seq_len}"
        self.fingerprint = self._build_fingerprint()
        self.index = None
        self.metadata: List[Dict] = []

    def _cache_paths(self) -> Tuple[Path, Path]:
        dim = self.model.get_sentence_embedding_dimension()
        if self.fingerprint is not None:
            prefix = f"{self.fingerprint.cache_key()}_text"
        else:
            prefix = f"{self.kb_hash}_{dim}_text"
        return self.cache_dir / f"{prefix}.faiss", self.cache_dir / f"{prefix}.pkl"

    def _build_fingerprint(self) -> Optional[IndexFingerprint]:
        if not self.snapshot_id:
            return None
        dim = int(self.model.get_sentence_embedding_dimension())
        return IndexFingerprint(
            snapshot_id=self.snapshot_id,
            embedding_model=self.model_name,
            embedding_revision=self.embedding_revision,
            embedding_dimension=dim,
            normalization_method="l2",
            distance_metric="ip",
            docir_schema_version=self.snapshot_config.get("docir_schema_version", 2),
            chunker_version=self.snapshot_config.get("chunker_version") or os.environ.get("FINANCE_CHUNKER_VERSION", "chunk-v2"),
            parser_versions_hash=self.snapshot_config.get("parser_versions_hash") or os.environ.get(
                "FINANCE_PARSER_VERSIONS_HASH",
                "parsers-v2",
            ),
            retrieval_channel="text",
            demo_subset_hash=self._demo_subset_hash(),
        )

    @staticmethod
    def _demo_subset_hash() -> str | None:
        raw = os.environ.get("FINANCE_DEMO_KEYWORDS") or ""
        if not raw:
            return None
        return hashlib.md5(raw.strip().encode("utf-8")).hexdigest()[:16]

    def build(self, blocks: List[Dict], force_rebuild: bool = False) -> None:
        index_path, meta_path = self._cache_paths()
        if not force_rebuild and index_path.exists() and meta_path.exists():
            saved_meta = load_index_fingerprint_meta(index_path)
            block_digest = None
            if self.fingerprint is not None:
                block_digest = _digest_block_ids(blocks)
                if not index_meta_matches(
                    self.fingerprint,
                    saved_meta,
                    block_ids_sha256=block_digest,
                ):
                    print(
                        f"Fingerprint 不匹配，重建文本缓存: {index_path.name}",
                        flush=True,
                    )
                    force_rebuild = True
            if not force_rebuild:
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
        if self.fingerprint is not None:
            write_index_fingerprint_meta(
                index_path,
                self.fingerprint,
                block_ids_sha256=_digest_block_ids(blocks),
            )
        print(f"文本向量已缓存: {index_path.name}", flush=True)

    def search(
        self,
        query: str,
        top_k: int = 10,
        visibility: Optional[Any] = None,
    ) -> List[Dict]:
        if self.index is None:
            return []
        candidate_k = top_k
        if visibility is not None:
            candidate_k = min(len(self.metadata), max(top_k * 4, top_k))
        q = self.model.encode([query], batch_size=1, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, candidate_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["score"] = float(score)
                item["retrieval_channel"] = "text"
                out.append(item)
        if visibility is not None:
            out = [
                item
                for item in out
                if visibility.allowed(item.get("document_version") or "")
            ][:top_k]
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
        snapshot_id: Optional[str] = None,
        snapshot_config: Optional[Dict] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_hash = kb_hash
        self.snapshot_id = snapshot_id
        self.snapshot_config = snapshot_config or {}
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
        if self.snapshot_id:
            prefix = f"{self._path_fingerprint().cache_key()}_image_page_identity_v2"
        else:
            prefix = f"{self.kb_hash}_{self.backend}_image_page_identity_v2"
        return self.cache_dir / f"{prefix}.faiss", self.cache_dir / f"{prefix}.pkl"

    def _path_fingerprint(self, dim: int = 0) -> IndexFingerprint:
        pipeline = getattr(self, "snapshot_config", {})
        return IndexFingerprint(
            snapshot_id=self.snapshot_id or self.kb_hash,
            embedding_model=self.model_name,
            embedding_revision="main",
            embedding_dimension=int(dim),
            normalization_method="l2",
            distance_metric="ip",
            docir_schema_version=pipeline.get("docir_schema_version", 2),
            chunker_version=pipeline.get("chunker_version") or os.environ.get("FINANCE_CHUNKER_VERSION", "chunk-v2"),
            parser_versions_hash=pipeline.get("parser_versions_hash") or os.environ.get(
                "FINANCE_PARSER_VERSIONS_HASH",
                "parsers-v2",
            ),
            retrieval_channel=f"image_{self.backend}",
        )

    def build(self, blocks: List[Dict], page_images: Dict[Tuple[str, int], str], force_rebuild: bool = False) -> None:
        index_path, meta_path = self._cache_paths()
        if not force_rebuild and index_path.exists() and meta_path.exists():
            if self.snapshot_id:
                saved_meta = load_index_fingerprint_meta(index_path)
                dim = (
                    int(saved_meta.get("embedding_dimension") or 0)
                    if isinstance(saved_meta, dict)
                    else 0
                )
                validation = self._path_fingerprint(dim=dim)
                if not index_meta_matches(
                    validation,
                    saved_meta,
                    block_ids_sha256=_digest_block_ids(blocks),
                ):
                    print(
                        f"图像指纹不匹配，重建缓存: {index_path.name}",
                        flush=True,
                    )
                    force_rebuild = True
            if not force_rebuild:
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
        if self.snapshot_id:
            write_index_fingerprint_meta(
                index_path,
                self._path_fingerprint(dim=int(embeddings.shape[1])),
                block_ids_sha256=_digest_block_ids(blocks),
            )

    @staticmethod
    def _build_page_items(blocks: List[Dict], page_images: Dict[Tuple[str, int], str]) -> List[Dict]:
        grouped: Dict[Tuple[str, int], List[Dict]] = {}
        for b in blocks:
            if b.get("page") is None:
                continue
            key = page_image_key(b, int(b["page"]), page_images)
            grouped.setdefault(key, []).append(b)

        items: List[Dict] = []
        for (doc, page), rows in grouped.items():
            image_ref = page_images.get((doc, page)) or rows[0].get("image_ref")
            items.append(
                {
                    "source_doc": rows[0]["source_doc"],
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

    def search(
        self,
        query: str,
        top_k: int = 6,
        visibility: Optional[Any] = None,
    ) -> List[Dict]:
        if self.index is None:
            return []
        candidate_k = top_k
        if visibility is not None:
            candidate_k = min(len(self.metadata), max(top_k * 4, top_k))
        q = self._encode_query(query).astype(np.float32)
        q = np.expand_dims(q, axis=0)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, candidate_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["score"] = float(score)
                out.append(item)
        if visibility is not None:
            out = [
                item
                for item in out
                if visibility.allowed(item.get("document_version") or "")
            ][:top_k]
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

    def search(
        self,
        query: str,
        top_k: int = 10,
        visibility: Optional[Any] = None,
    ) -> List[Dict]:
        text_results = self.text_retriever.search(
            query,
            top_k=max(12, top_k),
            visibility=visibility,
        )
        image_results = self.image_retriever.search(
            query,
            top_k=max(8, top_k // 2),
            visibility=visibility,
        )
        text_rrf = self._rrf_rank(text_results)
        image_rrf = self._rrf_rank(image_results)

        merged: Dict[str, Dict] = {}
        for item, is_text in [(x, True) for x in text_results] + [(x, False) for x in image_results]:
            bid = item["block_id"]
            if bid not in merged:
                merged[bid] = dict(item)
            merged[bid]["fusion_score"] = text_rrf.get(bid, 0.0) + image_rrf.get(bid, 0.0)
            merged[bid]["text_rrf"] = text_rrf.get(bid, 0.0)
            merged[bid]["image_rrf"] = image_rrf.get(bid, 0.0)
            if item.get("image_score") is not None:
                for key in ("image_score", "image_ref", "vision_page", "page_id", "image_sha256"):
                    merged[bid][key] = item.get(key)
            if is_text:
                merged[bid]["dense_score"] = float(item.get("dense_score", item.get("score")) or 0.0)
            channel = item.get("retrieval_channel")
            channels = merged[bid].setdefault("retrieval_channels", [])
            if channel and channel not in channels:
                channels.append(channel)

        ranked = sorted(merged.values(), key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        if visibility is not None:
            ranked = [
                item
                for item in ranked
                if visibility.allowed(item.get("document_version") or "")
            ]
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
    def __init__(
        self,
        fusion_retriever: FusionRetriever,
        graph_retriever: Optional[Neo4jGraphRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        structured_retriever: Optional[StructuredRetriever] = None,
        lightrag_retriever: Optional[Any] = None,
        graph_weight: float = 1.0,
    ):
        self.fusion_retriever = fusion_retriever
        self.graph_retriever = graph_retriever
        self.sparse_retriever = sparse_retriever
        self.structured_retriever = structured_retriever
        # 第 4 通道（阶段 B）：LightRAG 的实体+关系双索引。
        # 与 Neo4jGraphRetriever 的区别：后者是"关键词/数值共现"的轻量本地图，
        # 前者是 LLM 抽取的真实实体关系图 —— 两者可共存，也可单独开关。
        self.lightrag_retriever = lightrag_retriever
        # 图通道权重，便于消融实验（设为 0 等价于关闭该通道）
        self.graph_weight = float(graph_weight if graph_weight is not None else 1.0)

    def search(
        self,
        query: str,
        top_k: int = 10,
        visibility: Optional[Any] = None,
        allow_remote_graph: bool = True,
    ) -> List[Dict]:
        dense = self.fusion_retriever.search(query, top_k=top_k, visibility=visibility)
        candidate_k = min(len(dense) + 200, max(top_k * 4, 20))
        sparse = (
            self.sparse_retriever.search(
                query,
                top_k=candidate_k,
                visibility=visibility,
            )
            if self.sparse_retriever is not None
            else []
        )
        structured = (
            self.structured_retriever.search(
                query,
                top_k=candidate_k,
                visibility=visibility,
            )
            if self.structured_retriever is not None
            else []
        )
        graph = (
            self.graph_retriever.search(query, top_k=max(4, top_k // 2))
            if self.graph_retriever is not None
            else []
        )
        # 第 4 通道：LightRAG 图检索。它自身保证「服务不可达 → 返回空列表」，
        # 因此这里不需要 try/except，也不会因为图服务故障拖垮其余三路。
        lightrag = (
            self.lightrag_retriever.search(
                query, top_k=candidate_k, visibility=visibility
            )
            if self.lightrag_retriever is not None and (
                allow_remote_graph or not getattr(self.lightrag_retriever, "requires_remote_model", False)
            )
            else []
        )
        for item in dense:
            if "dense_score" not in item and item.get("image_score") is None:
                item["dense_score"] = float(item.get("score") or 0.0)
            channels = item.setdefault("retrieval_channels", [])
            channel = item.get("retrieval_channel")
            if channel and channel not in channels:
                channels.append(channel)
        merged = {x["block_id"]: x for x in dense}
        for item in sparse + structured + lightrag:
            bid = item["block_id"]
            if bid not in merged:
                merged[bid] = dict(item)
            else:
                if item.get("sparse_score") is not None:
                    merged[bid]["sparse_score"] = max(
                        float(merged[bid].get("sparse_score") or 0.0),
                        float(item["sparse_score"]),
                    )
                if item.get("structured_score") is not None:
                    merged[bid]["structured_score"] = max(
                        float(merged[bid].get("structured_score") or 0.0),
                        float(item["structured_score"]),
                    )
                if item.get("graph_score") is not None:
                    merged[bid]["graph_score"] = max(
                        float(merged[bid].get("graph_score") or 0.0),
                        float(item["graph_score"]),
                    )
                if item.get("kg_entities"):
                    merged[bid]["kg_entities"] = item["kg_entities"]
                if item.get("kg_relations"):
                    merged[bid]["kg_relations"] = item["kg_relations"]
            channel = item.get("retrieval_channel")
            channels = merged[bid].setdefault("retrieval_channels", [])
            if channel and channel not in channels:
                channels.append(channel)
        for g in graph:
            if g["block_id"] not in merged:
                merged[g["block_id"]] = g
        dense_rrf = FusionRetriever._rrf_rank(dense)
        sparse_rrf = FusionRetriever._rrf_rank(sparse)
        structured_rrf = FusionRetriever._rrf_rank(structured)
        graph_rrf = FusionRetriever._rrf_rank(graph)
        lightrag_rrf = FusionRetriever._rrf_rank(lightrag)
        for item in merged.values():
            bid = item["block_id"]
            item["dense_rrf"] = dense_rrf.get(bid, 0.0)
            item["sparse_rrf"] = sparse_rrf.get(bid, 0.0)
            item["structured_rrf"] = structured_rrf.get(bid, 0.0)
            item["graph_rrf"] = graph_rrf.get(bid, 0.0)
            # 图通道权重可调：0 表示消融实验中关闭该路
            item["lightrag_rrf"] = lightrag_rrf.get(bid, 0.0) * self.graph_weight
            item["fusion_score"] = (
                item["dense_rrf"]
                + item["sparse_rrf"]
                + item["structured_rrf"]
                + item["graph_rrf"]
                + item["lightrag_rrf"]
            )
        ranked = sorted(
            merged.values(),
            key=lambda x: x.get("fusion_score", 0.0),
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item["fusion_rank"] = rank
        if visibility is not None:
            ranked = [
                item
                for item in ranked
                if visibility.allowed(item.get("document_version") or "")
            ]
        return ranked[:top_k]


class _EmptyImageRetriever:
    """演示/低内存模式：跳过图像塔，避免再加载一套向量模型。

    注意：`search` 必须接受 `visibility` 等关键字参数。历史上这里签名与
    `FusionRetriever` 的调用不一致，导致 skip_image 模式下整条融合检索
    抛 TypeError 并返回 0 条证据（静默失败，极难发现）。
    用 `**kwargs` 兜底，避免今后再加参数时重演。
    """

    def search(self, query: str, top_k: int = 6, **kwargs: Any) -> List[Dict]:
        _ = query, top_k, kwargs
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


def _digest_block_ids(blocks: List[Dict]) -> str:
    digest = hashlib.sha256()
    for block_id in sorted(str(block.get("block_id") or "") for block in blocks):
        digest.update(block_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


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
    explicit_doc_keywords = doc_keywords is not None
    doc_keywords = doc_keywords if doc_keywords is not None else _env_demo_keywords()
    skip_image = _env_skip_image() if skip_image is None else skip_image
    chinese_vision = not skip_image and os.environ.get('FINANCE_VISION_BACKEND', image_backend) == 'chinese_clip'

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
    if not blocks and doc_keywords and not chinese_vision and not explicit_doc_keywords:
        print("关键词未命中文档，回退为仅按 FINANCE_MAX_BLOCKS 截断加载…", flush=True)
        subset_tag = hashlib.md5(f"{max_blocks}|FALLBACK".encode("utf-8")).hexdigest()[:10]
        kb_hash = f"{base_hash}_demo{subset_tag}"
        blocks = store.load(max_blocks=max_blocks or 8000, doc_keywords=None)
    print(f"已加载文本块: {len(blocks)}", flush=True)
    visual_pages = None
    manifest_path = kb_file.parent / 'kb_v3_manifest.json'
    if chinese_vision and manifest_path.is_file():
        from .visual_pages import build_visual_pages
        visual_pages = build_visual_pages(manifest_path, blocks,
            allowed_roots=(kb_file.parent, Path(__file__).resolve().parents[2] / 'output/kb_v3/parsed'),
            selected_versions={b['document_version'] for b in blocks}
                if doc_keywords or (subset_tag and blocks) else None,
            source_keywords=doc_keywords)
    if not blocks and not visual_pages:
        raise ValueError(
            "精简加载后知识块为 0。请放宽 FINANCE_DEMO_KEYWORDS 或增大 FINANCE_MAX_BLOCKS。"
        )

    embed_device = resolve_embed_device()
    snapshot_id = blocks[0].get("snapshot_id") if blocks else None
    snapshot_config = {}
    if snapshot_id and manifest_path.is_file():
        from dataclasses import asdict
        from src.pipeline.snapshots import read_manifest
        manifest = read_manifest(manifest_path)
        if manifest.snapshot_id != snapshot_id:
            raise ValueError("文本切块与索引快照身份不一致")
        snapshot_config = asdict(manifest.config)
    text = TextRetriever(
        model_name=text_model,
        cache_dir=cache_dir,
        kb_hash=kb_hash,
        device=embed_device,
        snapshot_id=snapshot_id,
        snapshot_config=snapshot_config,
    ) if blocks else _EmptyImageRetriever()
    if blocks:
        text.build(blocks, force_rebuild=force_rebuild)

    if skip_image:
        image = _EmptyImageRetriever()
        print("已跳过图像检索塔（FINANCE_SKIP_IMAGE）。", flush=True)
    elif os.environ.get("FINANCE_VISION_BACKEND", image_backend) == "chinese_clip":
        from .vision import ChineseVisionRetriever
        image = ChineseVisionRetriever(cache_dir=cache_dir, device=embed_device)
        image.build(blocks=blocks, page_images=store.page_images, force_rebuild=force_rebuild,
                    visual_pages=visual_pages)
    else:
        image_model = resolve_colpali() if image_backend == "colpali" else resolve_clip_model()
        image = VisionPageRetriever(
            cache_dir=cache_dir,
            kb_hash=kb_hash,
            backend=image_backend,
            model_name=image_model,
            text_proxy_model=text_model,
            device=embed_device,
            snapshot_id=snapshot_id,
            snapshot_config=snapshot_config,
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
    sparse = SparseRetriever(store.blocks)
    structured = StructuredRetriever(store.blocks)
    from .lightrag_channel import build_lightrag_retriever

    graph_channel = build_lightrag_retriever(store.blocks)
    if (os.getenv("FINANCE_KG_PILOT_ENABLED") or "").strip().lower() in {"1", "true", "yes"}:
        from scripts.build_budgeted_kg import DEFAULT_DB, KGCheckpoint
        from .kg_pilot_channel import KGPilotRetriever

        checkpoint = KGCheckpoint(Path(os.getenv("FINANCE_KG_PILOT_DB") or DEFAULT_DB),
                                  recover_interrupted=False)
        graph_channel = KGPilotRetriever(store.blocks, checkpoint)

    hybrid = HybridRetriever(
        fusion_retriever=fusion,
        graph_retriever=graph_retriever,
        sparse_retriever=sparse,
        structured_retriever=structured,
        lightrag_retriever=graph_channel,
    )
    return store, text, image, hybrid
