"""Agent 状态契约。

设计要点：
    1. **可观测优先**：每个节点都往 `trace` 追加一条 `NodeTrace`，
       前端据此展示「路由决策 → 评分 → 重试」的完整轨迹；
    2. **重试计数显式化**：`retry_retrieve` / `retry_generate` 进入状态，
       配合条件边防止无限循环（这是参考实现里用消息条数判断阈值的改进版）；
    3. **证据字段复用检索层契约**：Evidence 直接沿用
       `src/pipeline/chunker.py :: Chunk.to_retrieval_item()` 的字段，
       避免在 Agent 层再造一套数据结构。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class NodeTrace(TypedDict, total=False):
    """单个节点的执行轨迹（可观测的最小单元）。"""

    node: str
    elapsed_ms: int
    decision: Optional[str]
    detail: Dict[str, Any]


class Evidence(TypedDict, total=False):
    """检索证据：字段与 `Chunk.to_retrieval_item()` 保持一致。"""

    chunk_id: str
    block_ids: List[str]
    source_doc: str
    source_url: str
    document_id: str
    document_version: str
    snapshot_id: Optional[str]
    page: Optional[int]
    page_start: Optional[int]
    page_end: Optional[int]
    section_path: Optional[str]
    type: str
    content: str
    bbox: Optional[List[float]]
    char_len: int
    # 各通道分数
    dense_score: float
    sparse_score: float
    structured_score: float
    graph_score: float
    fusion_score: float
    rerank_score: float
    retrieval_channels: List[str]
    # 图通道附加
    kg_entities: List[str]
    kg_relations: List[str]
    # 本节点写入
    relevant: Optional[bool]
    grade_reason: str


Route = Literal["retrieval", "websearch", "direct", "clarify"]


class AgentState(TypedDict, total=False):
    """LangGraph 的共享状态。所有节点读写同一份，`trace` 用 add 归约。"""

    # ---- 输入 ----
    question: str
    run_id: str
    knowledge_snapshot_id: str
    execution_signature: str
    top_k: int
    use_paid_model: bool
    use_web_search: bool

    # ---- 路由 ----
    route: Route
    route_reason: str

    # ---- 规划 ----
    sub_questions: List[str]

    # ---- 检索 ----
    documents: List[Evidence]
    web_documents: List[Evidence]
    local_relevant_documents: List[Evidence]
    relevant_documents: List[Evidence]
    graded_count: int
    filtered_count: int
    web_searched: bool
    web_search_error: str
    retrieve_top_k: int

    # ---- 生成 ----
    generation: str
    generation_truncated: bool
    citations: List[int]

    # ---- 双层校验 ----
    grounded: Optional[bool]
    hallucination_reason: str
    answers_question: Optional[bool]
    answer_reason: str
    # 双层校验是否真的执行过。LLM 不可用/被关闭时两层都是「空过」，
    # 置信度不能因此给满分 —— 那是「没校验」而不是「校验通过」。
    verified: bool
    grounded_verified: bool
    answer_verified: bool

    # ---- 重试计数（条件边据此判定终止）----
    retry_retrieve: int
    retry_generate: int

    # ---- 收尾 ----
    status: str          # completed | no_evidence | direct | clarify
    confidence: float

    # ---- 可观测 ----
    trace: Annotated[List[NodeTrace], add]


def initial_state(
    question: str,
    *,
    run_id: str = "",
    knowledge_snapshot_id: str = "",
    top_k: int = 12,
    use_paid_model: bool = True,
    use_web_search: bool = True,
) -> AgentState:
    return AgentState(
        question=question,
        run_id=run_id,
        knowledge_snapshot_id=knowledge_snapshot_id,
        top_k=top_k,
        use_paid_model=use_paid_model,
        use_web_search=use_web_search,
        retrieve_top_k=top_k,
        documents=[],
        web_documents=[],
        local_relevant_documents=[],
        relevant_documents=[],
        web_searched=False,
        web_search_error="",
        sub_questions=[],
        citations=[],
        retry_retrieve=0,
        retry_generate=0,
        verified=False,
        grounded_verified=False,
        answer_verified=False,
        trace=[],
    )


# 条件边的默认阈值（可通过环境变量覆盖，见 nodes.py）
MAX_RETRY_RETRIEVE = 2
MAX_RETRY_GENERATE = 1
