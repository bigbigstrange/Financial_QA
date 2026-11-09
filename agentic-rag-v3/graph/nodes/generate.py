from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    # 计数放在生成节点：每真正生成一次记一次，重试上限由 graph 的条件边判断。
    return {
        **state,
        "documents": documents,
        "question": question,
        "generation": generation,
        "iterations": int(state.get("iterations", 0)) + 1,
    }
