"""LangGraph 图装配与对外入口。

图结构（节点名与 `state.py` 的 trace.node 保持一致，便于前端展示轨迹）：

    START → route_query ─┬─ direct_answer → END
                         ├─ clarify → END
                         └─ plan_subquestions → retrieve → grade_documents
                                ├─ 有相关 → generate → check_hallucination
                                │                ├─ 通过 → check_answer
                                │                │            ├─ 通过 → END
                                │                │            └─ 偏题 & 未达上限 → retrieve
                                │                └─ 幻觉 & 未达上限 → generate
                                ├─ 全无关 & 未达上限 → rewrite_query → retrieve
                                └─ 全无关 & 达上限 → give_up → END
"""
from __future__ import annotations

import os
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from langchain_core.runnables import RunnableConfig

from .edges import (
    after_route,
    make_after_answer_check,
    make_after_grade,
    make_after_hallucination,
)
from .nodes import AgentConfig, AgentNodes
from .state import AgentState, initial_state

END = "__end__"
DEFAULT_RECURSION_LIMIT = 32


def build_graph(
    retriever: Any,
    llm: Any = None,
    *,
    config: Optional[AgentConfig] = None,
    planner: Any = None,
    reranker: Any = None,
    checkpointer: Any = None,
    log: Optional[Any] = None,
    web_search: Any = None,
    node_journal: Any = None,
):
    """装配并编译 LangGraph 状态图。

    `llm=None` 时图仍可运行（路由退化为总是检索、校验直接通过、生成走抽取式），
    便于无 Key 环境与单元测试。
    """
    from langgraph.graph import END as LG_END
    from langgraph.graph import StateGraph

    config = config or AgentConfig.from_env()
    nodes = AgentNodes(retriever, llm, config=config, planner=planner,
                       reranker=reranker, log=log, web_search=web_search)
    text_fingerprint = getattr(getattr(getattr(retriever, "fusion_retriever", None),
                                      "text_retriever", None), "fingerprint", None)
    source_root = Path(__file__).resolve().parents[1]
    signature = hashlib.sha256(b"".join(
        (source_root / name).read_bytes() for name in (
            "agent/graph.py", "agent/nodes.py", "agent/prompts.py", "agent/edges.py",
            "agent/visual_answer.py", "system/agent.py", "system/retrieval.py",
            "system/rerank.py", "system/web_search.py", "system/llm_api.py",
            "models/gateway.py")) + json.dumps({
                "config": asdict(config), "provider": getattr(llm, "provider", None),
                "model": getattr(llm, "model", None), "thinking_disabled": getattr(llm, "disable_thinking", None),
                "text_index": text_fingerprint.to_dict() if text_fingerprint is not None else None,
                "web_domains": getattr(nodes.web_search_client, "allowed_domains", ()),
                "rerank_backend": getattr(nodes.reranker, "backend", None),
                "policies": {key: os.getenv(key, "") for key in (
                    "FINANCE_SKIP_IMAGE", "FINANCE_ENABLE_GRAPH_CHANNEL", "FINANCE_KG_PILOT_ENABLED", "FINANCE_PILOT_VISUAL",
                    "FINANCE_PILOT_VISUAL_PAGE_ID", "FINANCE_PILOT_VISUAL_SHA256")},
            }, sort_keys=True).encode()).hexdigest()

    def durable(name: str, operation):
        def execute(state: AgentState, config: RunnableConfig):
            from src.system.execution_control import check_execution
            check_execution()
            if node_journal is None or not state.get("run_id"):
                return operation(state)
            return node_journal.execute(state["run_id"], name,
                int(config.get("metadata", {}).get("langgraph_step", 0)),
                signature, state, lambda: operation(state))
        return execute

    graph = StateGraph(AgentState)
    for name in ("route_query", "plan_subquestions", "retrieve", "grade_documents",
                 "web_search", "rewrite_query", "generate", "check_hallucination",
                 "check_answer", "finalize", "direct_answer", "clarify", "give_up"):
        graph.add_node(name, durable(name, getattr(nodes, name)))

    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query", after_route,
        {"direct_answer": "direct_answer", "clarify": "clarify",
         "web_search": "web_search",
         "plan_subquestions": "plan_subquestions"},
    )
    graph.add_edge("plan_subquestions", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        make_after_grade(config.max_retry_retrieve,
                         web_available=nodes.web_search_client is not None),
        {"generate": "generate", "web_search": "web_search", "rewrite_query": "rewrite_query",
         "give_up": "give_up"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges("web_search",
                                lambda state: "grade_documents" if state.get("web_documents") else "give_up",
                                {"grade_documents": "grade_documents", "give_up": "give_up"})
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        make_after_hallucination(config.max_retry_generate),
        {"check_answer": "check_answer", "generate": "generate", "give_up": "give_up"},
    )
    graph.add_conditional_edges(
        "check_answer",
        make_after_answer_check(config.max_retry_retrieve),
        {"__end__": "finalize", "rewrite_query": "rewrite_query", "give_up": "give_up"},
    )
    graph.add_edge("finalize", LG_END)
    graph.add_edge("direct_answer", LG_END)
    graph.add_edge("clarify", LG_END)
    graph.add_edge("give_up", LG_END)

    compiled = graph.compile(checkpointer=checkpointer)
    compiled.execution_signature = signature
    return compiled


def run_agent(
    graph: Any,
    question: str,
    *,
    run_id: str = "",
    knowledge_snapshot_id: str = "",
    top_k: int = 12,
    use_paid_model: bool = True,
    use_web_search: bool = True,
    config: Optional[Dict[str, Any]] = None,
    on_node: Callable[[dict], None] | None = None,
) -> Dict[str, Any]:
    """同步执行一次 Agent 问答，返回最终状态（含 trace）。"""
    state = initial_state(
        question,
        run_id=run_id,
        knowledge_snapshot_id=knowledge_snapshot_id,
        top_k=top_k,
        use_paid_model=use_paid_model,
        use_web_search=use_web_search,
    )
    state["execution_signature"] = getattr(graph, "execution_signature", "")
    invoke_config = dict(config or {})
    invoke_config.setdefault("recursion_limit", DEFAULT_RECURSION_LIMIT)
    if run_id:
        configurable = dict(invoke_config.get("configurable") or {})
        configurable.setdefault("thread_id", run_id)
        invoke_config["configurable"] = configurable
    if on_node is None:
        return graph.invoke(state, invoke_config)
    seen = 0
    final = state
    for value in graph.stream(state, invoke_config, stream_mode="values"):
        final = value
        trace = list(value.get("trace") or [])
        for entry in trace[seen:]:
            on_node({"node": entry.get("node"), "decision": entry.get("decision"),
                     "elapsed_ms": entry.get("elapsed_ms")})
        seen = len(trace)
    return final


def build_and_run(
    retriever: Any,
    question: str,
    *,
    llm: Any = None,
    run_id: str = "",
    knowledge_snapshot_id: str = "",
    top_k: int = 12,
) -> Dict[str, Any]:
    graph = build_graph(retriever, llm)
    return run_agent(graph, question, run_id=run_id,
                     knowledge_snapshot_id=knowledge_snapshot_id, top_k=top_k)
