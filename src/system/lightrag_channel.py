"""LightRAG 图通道适配器（阶段 B · 第 4 检索通道）。

## 定位

与 `TextRetriever` / `SparseRetriever` / `StructuredRetriever` 并列的**第 4 路**，
由 `HybridRetriever` 统一按 RRF 融合。这样做的三个理由：

1. 可做**消融实验**：能回答"图通道到底贡献了多少指标"；
2. 可**单独降级**：LightRAG 服务挂了，其余三路照常工作；
3. 复用现有**分数契约**（`graph_rrf` / `fusion_score`），不引入新的融合逻辑。

## chunk_id 对齐

图证据（实体/关系）通过 `source_id` 指回文本 chunk。若对齐成功，
`graph_score` 能直接落到我们自己的 `chunk_id` 上；若 LightRAG 不保留我们的 id，
则退化为按 `(source_doc, page, section_path)` 在本地 `chunks.jsonl` 里反查 ——
精度略降但 RRF 仍可用。对齐能力由 `src/system/lightrag_client.py :: probe_capabilities()`
在部署后实测确认。
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .lightrag_client import LightRAGClient, LightRAGConfig, LightRAGError

# LightRAG 把多个来源拼在一个字符串里，分隔符在不同版本可能不同
_SOURCE_SEP = re.compile(r"[,\u3001;|]|<SEP>|\n")

# 各类型图证据的权重：实体最直接，关系次之，原文 chunk 作为兜底
_WEIGHT_ENTITY = 1.0
_WEIGHT_RELATION = 0.8
_WEIGHT_CHUNK = 0.5


class LightRAGRetriever:
    """第 4 检索通道：把 LightRAG 的实体/关系/文本块映射成统一 Evidence。"""

    requires_remote_model = True

    def __init__(
        self,
        blocks: Sequence[Dict[str, Any]],
        *,
        client: Optional[LightRAGClient] = None,
        config: Optional[LightRAGConfig] = None,
        mode: str = "mix",
        top_k: Optional[int] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.blocks = list(blocks)
        self.client = client or LightRAGClient(config)
        self.mode = (mode or "mix").strip().lower()
        self.top_k = top_k
        self.log = log or (lambda msg: print(msg, flush=True))

        # 主索引：chunk_id → 我们的检索单元（用于回填 source_doc/page/section）
        self.by_chunk_id: Dict[str, Dict[str, Any]] = {
            str(b.get("chunk_id") or b.get("block_id")): b
            for b in self.blocks
            if (b.get("chunk_id") or b.get("block_id"))
        }
        # 回退索引：图证据只给出文档名/页码时的模糊匹配
        self.by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for b in self.blocks:
            self.by_doc.setdefault(str(b.get("source_doc") or ""), []).append(b)

        self._available: Optional[bool] = None
        self.stats: Dict[str, int] = {
            "queries": 0, "errors": 0, "mapped": 0, "unmapped": 0, "fallback_mapped": 0,
        }

    # ---- 可用性 -----------------------------------------------------------

    @property
    def available(self) -> bool:
        """首次访问时探测一次，之后缓存结果（避免每个 query 都打健康检查）。"""
        if self._available is None:
            self._available = self.client.available()
            if not self._available:
                self.log(
                    f"[lightrag] 服务不可达（{self.client.config.base_url}），"
                    "图通道本轮禁用；其余检索通道不受影响。"
                )
        return self._available

    def reset_availability(self) -> None:
        """服务重启后调用，强制下次重新探测。"""
        self._available = None

    # ---- 检索 -------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        visibility: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """返回 [{block_id, chunk_id, content, graph_score, retrieval_channel, kg_*}]。

        **任何异常都不向上抛**：图通道是增强项，失败必须静默降级，
        不能让 LightRAG 的问题变成整个问答失败。
        """
        if not self.available:
            return []
        limit = int(top_k or self.top_k or 10)
        self.stats["queries"] += 1
        try:
            payload = self.client.query_data(
                query, mode=self.mode, top_k=limit,
                include_chunk_content=True,
            )
        except LightRAGError as exc:
            self.stats["errors"] += 1
            self.log(f"[lightrag] 检索失败，本轮回空: {exc}")
            return []
        except Exception as exc:  # noqa: BLE001 — 兜底，绝不外抛
            self.stats["errors"] += 1
            self.log(f"[lightrag] 未预期错误，本轮回空: {type(exc).__name__}: {exc}")
            return []

        try:
            return self._to_evidence(payload, limit, visibility=visibility)
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            self.log(f"[lightrag] 结果映射失败，本轮回空: {type(exc).__name__}: {exc}")
            return []

    # ---- 映射 -------------------------------------------------------------

    @staticmethod
    def _split_sources(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            out: List[str] = []
            for item in raw:
                out.extend(LightRAGRetriever._split_sources(item))
            return out
        return [p.strip() for p in _SOURCE_SEP.split(str(raw)) if p.strip()]

    @staticmethod
    def _bump(scores: Dict[str, float], key: str, weight: float) -> None:
        scores[key] = scores.get(key, 0.0) + weight

    def _collect(self, payload: Dict[str, Any]) -> Tuple[
        Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]
    ]:
        scores: Dict[str, float] = {}
        entities: Dict[str, List[str]] = {}
        relations: Dict[str, List[str]] = {}

        def _iter(key: str):
            value = payload.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
            return []

        for ent in _iter("entities"):
            name = str(ent.get("entity_name") or ent.get("name") or "").strip()
            for src in self._split_sources(
                ent.get("source_id") or ent.get("source_ids") or ent.get("chunk_id")
            ):
                self._bump(scores, src, _WEIGHT_ENTITY)
                if name:
                    entities.setdefault(src, []).append(name)

        for rel in _iter("relations"):
            head = str(rel.get("src_id") or rel.get("head") or "").strip()
            tail = str(rel.get("tgt_id") or rel.get("tail") or "").strip()
            label = f"{head}→{tail}" if head or tail else ""
            for src in self._split_sources(
                rel.get("source_id") or rel.get("source_ids") or rel.get("chunk_id")
            ):
                self._bump(scores, src, _WEIGHT_RELATION)
                if label:
                    relations.setdefault(src, []).append(label)

        for ch in _iter("chunks"):
            cid = ch.get("chunk_id") or ch.get("id") or ch.get("chunk_key")
            if cid:
                self._bump(scores, str(cid), _WEIGHT_CHUNK)

        return scores, entities, relations

    def _fallback_by_locator(self, source_key: str) -> Optional[Dict[str, Any]]:
        """chunk_id 对不上时，按「文档名 + 页码」在本地 chunks 里退化匹配。"""
        match = re.search(r"(.+?)[#:]p?(\d+)", source_key)
        doc_hint = match.group(1) if match else source_key
        page_hint = int(match.group(2)) if match else None
        for doc, items in self.by_doc.items():
            if not doc or doc not in doc_hint:
                continue
            if page_hint is None:
                return None
            for item in items:
                page = item.get("page_start", item.get("page"))
                if page == page_hint:
                    return item
            return None
        return None

    def _align_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Resolve server chunks by source AND original text; never invent a page.

        The server re-chunks documents and generates its own IDs. Only accept
        an unambiguous substantial text containment in the named local source.
        A document-only match is insufficient evidence for a particular chunk.
        """
        source = str(chunk.get("file_path") or "").replace("\\", "/")
        candidates = [b for doc, rows in self.by_doc.items()
                      if source == doc or source.endswith("/" + doc)
                      for b in rows]
        text = re.sub(r"\s+", "", str(chunk.get("content") or ""))
        exact = [b for b in candidates if text and
                 re.sub(r"\s+", "", str(b.get("content") or "")) == text]
        if len(exact) == 1:
            return exact[0]
        if exact:
            return None
        if len(text) < 40:
            return None
        matches = []
        for block in candidates:
            local = re.sub(r"\s+", "", str(block.get("content") or ""))
            if len(local) >= 40 and (local in text or text in local):
                matches.append(block)
        return matches[0] if len(matches) == 1 else None

    def _to_evidence(
        self,
        payload: Dict[str, Any],
        top_k: int,
        *,
        visibility: Optional[Any],
    ) -> List[Dict[str, Any]]:
        payload = payload.get("data", payload)
        if not isinstance(payload, dict):
            return []
        aligned = {}
        for chunk in payload.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            source_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            if source_id:
                block = self._align_chunk(chunk)
                if block is not None:
                    aligned[source_id] = block
        scores, entities, relations = self._collect(payload)
        if not scores:
            return []

        # 归一化到 0~1，便于与其它通道的分数并列观察（RRF 只用排名，不受影响）
        peak = max(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for source_key, raw_score in ranked:
            if len(out) >= top_k:
                break
            base = self.by_chunk_id.get(source_key)
            mapped_exactly = base is not None
            if base is None:
                base = aligned.get(source_key) or self._fallback_by_locator(source_key)
                if base is not None:
                    self.stats["fallback_mapped"] += 1
            if base is None:
                self.stats["unmapped"] += 1
                continue

            chunk_id = str(base.get("chunk_id") or base.get("block_id"))
            if chunk_id in seen:
                continue
            if visibility is not None and not visibility.allowed(
                base.get("document_version") or ""
            ):
                continue
            seen.add(chunk_id)
            if mapped_exactly:
                self.stats["mapped"] += 1

            item = dict(base)
            item["block_id"] = chunk_id
            item["chunk_id"] = chunk_id
            item["graph_score"] = round(float(raw_score) / peak, 6)
            item["graph_raw_score"] = round(float(raw_score), 4)
            item["retrieval_channel"] = "graph_lightrag"
            item["retrieval_channels"] = ["graph_lightrag"]
            item["kg_entities"] = entities.get(source_key, [])[:8]
            item["kg_relations"] = relations.get(source_key, [])[:8]
            item["kg_source_key"] = source_key
            item["kg_mapped_exactly"] = mapped_exactly
            out.append(item)
        return out


def build_lightrag_retriever(
    blocks: Sequence[Dict[str, Any]],
    *,
    enabled: Optional[bool] = None,
    **kwargs: Any,
) -> Optional[LightRAGRetriever]:
    """按环境变量装配图通道。

    `FINANCE_ENABLE_GRAPH_CHANNEL=0`（默认）时返回 None，
    即**默认关闭**，验证通过后再打开 —— 与总方案 §4.4 的开关约定一致。
    """
    import os
    from .cost_policy import paid_api_disabled

    if paid_api_disabled():
        return None

    if enabled is None:
        raw = (os.environ.get("FINANCE_ENABLE_GRAPH_CHANNEL") or "0").strip().lower()
        enabled = raw in {"1", "true", "yes", "y"}
    if not enabled:
        return None
    return LightRAGRetriever(blocks, **kwargs)
