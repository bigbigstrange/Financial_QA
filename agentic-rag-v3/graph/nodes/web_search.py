from typing import Any, Dict

from graph.state import GraphState


class ExternalSearchDisabled(RuntimeError):
    """Raised when a graph requests external search without a real search provider."""


def web_search(state: GraphState) -> Dict[str, Any]:
    question = state.get("question", "")
    raise ExternalSearchDisabled(
        f"External web search is disabled; no verified provider is configured for: {question}"
    )
