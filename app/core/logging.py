# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    tid = trace_id_var.get()
    if not tid:
        tid = uuid.uuid4().hex[:12]
        trace_id_var.set(tid)
    return tid


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | trace=%(trace_id)s | %(name)s | %(message)s",
    )

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.trace_id = trace_id_var.get() or "-"
        return record

    logging.setLogRecordFactory(record_factory)
