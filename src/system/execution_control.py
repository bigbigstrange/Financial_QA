"""Propagate a run's cancellation/deadline guard to node and model boundaries."""
from contextvars import ContextVar

execution_guard = ContextVar("finance_execution_guard", default=None)


def check_execution() -> None:
    guard = execution_guard.get()
    if guard is not None:
        guard()
