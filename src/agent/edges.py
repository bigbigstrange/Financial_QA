"""Agent 的条件边判定。

两条纠正路径**必须分开**（这正是双层校验的价值）：
    - 幻觉问题（`grounded=False`）→ 回 `generate`：用同一批证据重新生成。
      证据本身没问题，是生成跑偏了，换信息源没有意义。
    - 偏题问题（`answers_question=False`）→ 回 `retrieve`：换检索参数与信息源。
      重复使用同一批文档只会得到同样的偏题答案。

两个重试计数各自独立、各有硬上限。上限通过**闭包绑定**而不是写进 state，
避免污染状态契约（state 是给 LangGraph 与前端共用的公开结构）。
"""
from __future__ import annotations

from typing import Callable

from .state import AgentState, MAX_RETRY_GENERATE, MAX_RETRY_RETRIEVE

EdgeFn = Callable[[AgentState], str]


def after_route(state: AgentState) -> str:
    route = (state.get("route") or "retrieval").lower()
    if route == "direct":
        return "direct_answer"
    if route == "clarify":
        return "clarify"
    if route == "websearch":
        return "web_search"
    return "plan_subquestions"


def make_after_grade(max_retry_retrieve: int = MAX_RETRY_RETRIEVE,
                     web_available: bool = False) -> EdgeFn:
    def _edge(state: AgentState) -> str:
        if state.get("relevant_documents"):
            return "generate"
        if state.get("web_searched"):
            return "give_up"
        if web_available and state.get("use_web_search", True) and not state.get("web_searched"):
            return "web_search"
        if int(state.get("retry_retrieve") or 0) < max_retry_retrieve:
            return "rewrite_query"
        return "give_up"

    return _edge


def make_after_hallucination(max_retry_generate: int = MAX_RETRY_GENERATE) -> EdgeFn:
    """幻觉分支：用同一批证据重新生成。

    `retry_generate` 记录**校验不通过次数**（不是"已重试次数"）：
    未通过且 `retry_generate <= max_retry_generate` 时允许再生成一次，否则放弃。
    因此 `max_retry_generate=N` 表示"最多容忍 N 次不通过"，生成最多被调用 N+1 次。
    """

    def _edge(state: AgentState) -> str:
        if state.get("grounded", True):
            return "check_answer"
        if int(state.get("retry_generate") or 0) <= max_retry_generate:
            return "generate"
        return "give_up"

    return _edge


def make_after_answer_check(max_retry_retrieve: int = MAX_RETRY_RETRIEVE) -> EdgeFn:
    """偏题分支：**必须回到 rewrite_query**。

    如果直接回 `retrieve`，`retry_retrieve` 永远不会递增，
    条件边会一直放行 → 图无限循环直到 GraphRecursionError。
    经 `rewrite_query` 走一圈既做了查询改写（换信息源），也让计数器生效。
    """

    def _edge(state: AgentState) -> str:
        if state.get("answers_question", True):
            return "__end__"
        if state.get("web_searched"):
            return "give_up"
        if int(state.get("retry_retrieve") or 0) < max_retry_retrieve:
            return "rewrite_query"
        return "give_up"

    return _edge


# 默认上限的便捷实例（未显式配置时使用）
after_grade = make_after_grade()
after_hallucination = make_after_hallucination()
after_answer_check = make_after_answer_check()
