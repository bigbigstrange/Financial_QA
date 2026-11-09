"""LightRAG Server 的轻量 REST 客户端（阶段 B）。

## 为什么走 REST 而不是 SDK

LightRAG 官方 README 明确写着：

> For integration into your project, we strongly recommend using the **REST API**
> provided by the LightRAG Server. The LightRAG SDK is primarily intended for
> embedded applications or academic research and evaluation purposes.

因此本项目把 `lightrag-server` 当作**独立服务**部署，通过 HTTP 调用。
好处是知识图谱的构建/更新不占用本应用的进程与内存，也避免与 torch 版本冲突。

## 端点契约需要现场校验

LightRAG 迭代很快（2026.05 起把 RAG-Anything 合并进主干，还引入了多模态解析、
四种切块策略、按角色的 4 个 LLM 配置）。本文件里的端点路径与字段名**基于官方文档**，
但不同版本的字段可能有差异，因此：

1. 所有路径与字段名都可用环境变量覆盖（见 `LightRAGConfig`）；
2. 提供 `probe_capabilities()` —— 部署后**第一件事就是跑它**（对应总方案 §4.3 的
   chunk_id 对齐可行性验证 B3）。它会把探测结果写成报告，明确告诉我们
   "自定义 chunk_id 能否原样保留"，这决定了图证据能否精确映射回文本 chunk。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:9621"


@dataclass
class LightRAGConfig:
    """LightRAG 服务连接与端点配置（全部可用环境变量覆盖）。"""

    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 1.5

    # 端点路径（不同版本可能不同，便于覆盖）
    path_health: str = "/health"
    path_query: str = "/query"
    path_query_data: str = "/query/data"
    path_insert_text: str = "/documents/text"
    path_insert_texts: str = "/documents/texts"
    path_delete_doc: str = "/documents/delete_document"
    path_clear_documents: str = "/documents"
    path_status_counts: str = "/documents/status_counts"
    path_pipeline_status: str = "/documents/pipeline_status"
    path_graph_labels: str = "/graph/label/list"

    # 插入时用于携带自定义 chunk_id 的字段名（按优先级尝试）
    chunk_id_fields: List[str] = field(default_factory=lambda: [
        "chunk_id", "id", "chunk_key", "source_id",
    ])

    @classmethod
    def from_env(cls) -> "LightRAGConfig":
        def _env(name: str, default: str = "") -> str:
            return (os.environ.get(name) or "").strip() or default

        return cls(
            base_url=_env("FINANCE_LIGHTRAG_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=_env("FINANCE_LIGHTRAG_API_KEY") or None,
            timeout=float(_env("FINANCE_LIGHTRAG_TIMEOUT", "60") or 60),
            max_retries=int(_env("FINANCE_LIGHTRAG_RETRIES", "2") or 2),
            path_health=_env("FINANCE_LIGHTRAG_PATH_HEALTH", "/health"),
            path_query=_env("FINANCE_LIGHTRAG_PATH_QUERY", "/query"),
            path_query_data=_env("FINANCE_LIGHTRAG_PATH_QUERY_DATA", "/query/data"),
            path_insert_text=_env("FINANCE_LIGHTRAG_PATH_INSERT", "/documents/text"),
            path_delete_doc=_env("FINANCE_LIGHTRAG_PATH_DELETE", "/documents/delete_document"),
        )


class LightRAGError(RuntimeError):
    """LightRAG 调用失败。调用方应捕获并降级（图通道不可用不能拖垮主链路）。"""


class LightRAGClient:
    """最小可用的 LightRAG HTTP 客户端。

    设计原则：**任何失败都抛 `LightRAGError`，由上层决定降级**，
    不在这里吞异常 —— 否则会重演 `_EmptyImageRetriever` 那种"静默返回 0 条"的故障。
    """

    def __init__(self, config: Optional[LightRAGConfig] = None) -> None:
        self.config = config or LightRAGConfig.from_env()
        self._headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            self._headers["X-API-Key"] = self.config.api_key

    # ---- 低层 -------------------------------------------------------------

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        import requests
        from .cost_policy import paid_api_disabled

        paid_paths = {self.config.path_query, self.config.path_query_data,
                      self.config.path_insert_text, self.config.path_insert_texts}
        if method.upper() == "POST" and path in paid_paths and paid_api_disabled():
            raise LightRAGError("付费图谱调用已暂停：FINANCE_DISABLE_PAID_API=1")

        url = f"{self.config.base_url}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = requests.request(
                    method, url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.config.timeout,
                )
                if resp.status_code >= 400:
                    raise LightRAGError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                if not resp.content:
                    return {}
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff ** attempt)
        raise LightRAGError(f"{method} {path} 失败: {type(last_err).__name__}: {last_err}")

    # ---- 端点 -------------------------------------------------------------

    def health(self) -> dict:
        """健康检查。返回 {} 或含状态字段的字典；不可用抛 LightRAGError。"""
        return self._request("GET", self.config.path_health)

    def available(self) -> bool:
        """可用性探测（不抛异常，供装配阶段决定是否启用图通道）。"""
        try:
            self.health()
            return True
        except Exception:  # noqa: BLE001
            return False

    def query(
        self,
        question: str,
        *,
        mode: str = "mix",
        top_k: int = 10,
        only_need_context: bool = True,
        enable_rerank: bool = False,
    ) -> dict:
        """检索（默认只要上下文，不让 LightRAG 生成答案 —— 生成由我们的 Agent 统一负责）。

        `mode`：local | global | hybrid | naive | mix（默认 mix = local+global+naive）
        `only_need_context=True` 时返回实体/关系/文本块，不返回 LLM 生成的回答，
        避免两套生成逻辑打架、也省 token。

        注意：`response` 字段是**字符串**（内嵌 Markdown JSON 代码块），
        需要结构化数据请用 `query_data()`。
        """
        payload = {
            "query": question,
            "mode": mode,
            "top_k": top_k,
            "only_need_context": only_need_context,
            "enable_rerank": enable_rerank,
        }
        return self._request("POST", self.config.path_query, payload)

    def query_data(
        self,
        question: str,
        *,
        mode: str = "mix",
        top_k: int = 10,
        include_chunk_content: bool = True,
    ) -> dict:
        """结构化检索：返回 `{status, message, data, metadata}`。

        `data` 里是**结构化的 entities / relations / chunks**，
        比解析 `/query` 的 response 字符串可靠得多 —— 这是图通道适配器应走的路。
        """
        payload = {
            "query": question,
            "mode": mode,
            "top_k": top_k,
            "only_need_context": True,
            "include_references": True,
            "include_chunk_content": include_chunk_content,
            "enable_rerank": False,
        }
        return self._request("POST", self.config.path_query_data, payload)

    def insert_texts(
        self,
        texts: List[str],
        *,
        file_sources: Optional[List[str]] = None,
    ) -> dict:
        """批量插入多份文档文本（比逐份 insert_text 少很多次往返）。

        `file_sources` 是每份文本的文档身份，长度须与 `texts` 一致。
        LightRAG 会自己对每份文本切块，因此**不要在同一个 file_source 下分多次插入**。
        """
        payload: Dict[str, Any] = {"texts": list(texts)}
        if file_sources:
            payload["file_sources"] = list(file_sources)
        return self._request("POST", self.config.path_insert_texts, payload)

    def clear_documents(self) -> dict:
        """清空全部文档及其派生的图谱/向量数据。"""
        return self._request("DELETE", self.config.path_clear_documents)

    def delete_documents(
        self,
        doc_ids: List[str],
        *,
        delete_llm_cache: bool = False,
    ) -> dict:
        """按 id 批量删除文档。

        注意请求体字段是 `doc_ids`（数组）—— 传单个 `doc_id` 会 422。
        `delete_llm_cache=False` 保留抽取缓存，重跑同一批文档时不重复计费。
        """
        return self._request(
            "DELETE",
            self.config.path_delete_doc,
            {"doc_ids": list(doc_ids), "delete_llm_cache": delete_llm_cache},
        )

    def status_counts(self) -> dict:
        return self._request("GET", self.config.path_status_counts)

    def graph_labels(self) -> list:
        data = self._request("GET", self.config.path_graph_labels)
        return data if isinstance(data, list) else (data.get("data") or [])

    def insert_text(
        self,
        text: str,
        *,
        chunk_id: Optional[str] = None,
        file_source: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """插入一段文本（我们自己的 chunk）。

        **关键点**：带上我们自己的 `chunk_id`，这样 LightRAG 抽出的实体/关系的
        `source_id` 就能直接映射回文本 chunk，图证据才能参与 RRF 融合。
        是否真的被保留需要 `probe_capabilities()` 验证（总方案 §4.3 / B3）。
        """
        payload: Dict[str, Any] = {"text": text}
        if file_source:
            payload["file_source"] = file_source
        if chunk_id:
            payload["chunk_id"] = chunk_id
        if extra:
            payload.update(extra)
        return self._request("POST", self.config.path_insert_text, payload)

    def delete_document(self, doc_id: str) -> dict:
        return self.delete_documents([doc_id])


# ---------------------------------------------------------------------------
# 能力探测（部署后的第一件事）
# ---------------------------------------------------------------------------

def probe_capabilities(
    client: LightRAGClient,
    *,
    sample_text: Optional[str] = None,
    sample_chunk_id: str = "ck_probe_0001",
) -> Dict[str, Any]:
    """探测部署好的 LightRAG 是否满足我们的集成要求。

    最重要的一项是 **chunk_id 对齐**：LightRAG 默认用自己的切块器，
    如果我们灌入的 `chunk_id` 不能被保留，图证据就只能按 (文档, 页码) 粗对齐，
    RRF 融合精度会下降。这个探测给出明确结论，而不是靠猜。
    """
    import sys
    from pathlib import Path

    report: Dict[str, Any] = {
        "base_url": client.config.base_url,
        "reachable": False,
        "health": None,
        "query_ok": False,
        "query_keys": [],
        "insert_ok": False,
        "chunk_id_preserved": None,
        "chunk_id_field": None,
        "conclusion": "",
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        report["health"] = client.health()
        report["reachable"] = True
    except LightRAGError as exc:
        report["conclusion"] = f"服务不可达：{exc}"
        return report

    # 1) 检索接口是否返回实体/关系/文本块
    try:
        data = client.query("测试检索连通性", mode="mix", top_k=3)
        report["query_keys"] = sorted(data.keys())
        report["query_ok"] = True
    except LightRAGError as exc:
        report["conclusion"] = f"query 接口不可用：{exc}"
        return report

    # 2) 插入一条带自定义 id 的文本，看 id 是否被保留
    text = sample_text or (
        "[章节] 探针/连通性测试\n"
        "同业业务是指以金融同业为交易对手方，同业资金融通等以我为目标的各类业务。\n"
        f"唯一标记：{sample_chunk_id}"
    )
    try:
        client.insert_text(text, chunk_id=sample_chunk_id,
                           file_source="__probe__/lightrag_alignment.txt")
        report["insert_ok"] = True
    except LightRAGError as exc:
        report["conclusion"] = f"insert 接口不可用：{exc}"
        return report

    # 3) 回查：用唯一标记检索，看返回的 chunk 标识是否等于我们给的 chunk_id
    try:
        back = client.query(sample_chunk_id, mode="naive", top_k=5)
    except LightRAGError:
        back = {}
    ids_seen: List[str] = []
    for key in ("chunks", "data", "results"):
        for item in (back.get(key) or []) if isinstance(back.get(key), list) else []:
            if isinstance(item, dict):
                for f in ("chunk_id", "id", "chunk_key"):
                    if item.get(f):
                        ids_seen.append(str(item[f]))
    if sample_chunk_id in ids_seen:
        report["chunk_id_preserved"] = True
        report["chunk_id_field"] = "chunk_id"
        report["conclusion"] = "✅ chunk_id 可原样保留，图证据可精确映射回文本 chunk"
    else:
        report["chunk_id_preserved"] = False
        report["conclusion"] = (
            "⚠️ 未能确认 chunk_id 被保留。回退方案：图证据按 (source_doc, page, "
            "section_path) 反向查我们的 chunks.jsonl —— 精度略降但 RRF 仍可用。"
            f"（回查看到的标识: {ids_seen[:5]}）"
        )

    out_dir = Path(os.environ.get("FINANCE_LIGHTRAG_PROBE_OUT")
                   or (Path.cwd() / "output" / "kb_v3"))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "lightrag_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    return report


if __name__ == "__main__":  # python -m src.system.lightrag_client
    cfg = LightRAGConfig.from_env()
    cli = LightRAGClient(cfg)
    print(json.dumps(probe_capabilities(cli), ensure_ascii=False, indent=2))
