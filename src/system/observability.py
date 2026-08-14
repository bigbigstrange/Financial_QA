import json
import logging
from typing import Any, Dict

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover
    Counter = None
    Histogram = None


LOGGER_NAME = "financial_qa"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(event: str, **kwargs: Dict[str, Any]) -> None:
    payload = {"event": event, **kwargs}
    get_logger().info(json.dumps(payload, ensure_ascii=False))


if Counter and Histogram:
    QUERY_COUNTER = Counter("qa_query_total", "Total query count", ["route"])
    QUERY_LATENCY = Histogram("qa_query_latency_seconds", "Query latency seconds", ["route"])
else:  # pragma: no cover
    QUERY_COUNTER = None
    QUERY_LATENCY = None
