from typing import Any, Dict

from graph.state import GraphState

REFUSAL_MESSAGE = (
    "根据现有资料无法给出可被证据支持的答案（多次重试后仍未通过依据校验），"
    "已明确拒答，请补充资料或联系人工核实。"
)


def give_up(state: GraphState) -> Dict[str, Any]:
    """重试达到上限仍不可用时终止，并给出明确拒答而不是硬编答案。"""
    print("---GIVE UP: NOT GROUNDED AFTER MAX_ITERATIONS---")
    return {**state, "generation": REFUSAL_MESSAGE, "refused": True}
