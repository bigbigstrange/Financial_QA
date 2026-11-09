from typing import List, TypedDict


class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        iterations: 已执行的生成次数（用于限制自纠正重试）
        refused: 是否因证据不足而明确拒答
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    iterations: int
    refused: bool
