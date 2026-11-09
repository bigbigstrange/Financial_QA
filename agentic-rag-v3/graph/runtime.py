"""运行期保护：整体超时 + 显式 recursion_limit。

图本身限制了重试次数（MAX_ITERATIONS），但模型/检索可能卡住，
所以在最外层再加一道墙钟超时，保证请求不会无限挂起。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.graph.state import CompiledStateGraph

DEFAULT_TIMEOUT_SECONDS = 30.0
RECURSION_LIMIT = 8


class GraphTimeoutError(TimeoutError):
    """整条图执行超过墙钟上限。"""


def invoke_with_timeout(
    graph: CompiledStateGraph,
    inputs: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    recursion_limit: int = RECURSION_LIMIT,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在独立线程里执行图，超时即放弃并抛出 GraphTimeoutError。"""
    run_config = dict(config or {})
    run_config["recursion_limit"] = recursion_limit

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(graph.invoke, inputs, run_config)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:  # concurrent.futures.TimeoutError
            raise GraphTimeoutError(
                f"graph execution exceeded {timeout}s (recursion_limit={recursion_limit})"
            ) from exc


def stream_with_timeout(
    graph: CompiledStateGraph,
    inputs: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    recursion_limit: int = RECURSION_LIMIT,
    config: dict[str, Any] | None = None,
):
    """流式版本的超时保护；逐块产出，超时抛 GraphTimeoutError。"""
    run_config = dict(config or {})
    run_config["recursion_limit"] = recursion_limit

    with ThreadPoolExecutor(max_workers=1) as pool:
        iterator = pool.submit(lambda: list(graph.stream(inputs, run_config))).result(
            timeout=timeout
        )
    yield from iterator
