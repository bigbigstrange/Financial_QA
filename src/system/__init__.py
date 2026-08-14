from __future__ import annotations

__all__ = ["FinancialQASystem"]


def __getattr__(name: str):  # PEP 562: 延迟导入，避免仅 import src.system.model_settings 时拉全链路
    if name == "FinancialQASystem":
        from .qa_engine import FinancialQASystem

        return FinancialQASystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
