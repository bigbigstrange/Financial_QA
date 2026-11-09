"""Agent 编排层：LangGraph 状态机（路由 → 检索 → 评分 → 分支 → 生成 → 双层校验）。

与 `src/system/agent.py`（legacy 的轻量 DocumentAgent）的关系：
    本模块是阶段 C 的新链路，`src/system/agent.py` 保留为可回退的 legacy 实现，
    通过环境变量 `FINANCE_AGENT_MODE=graph|legacy` 切换。
"""
from __future__ import annotations

from .graph import build_graph, run_agent
from .state import AgentState, NodeTrace

__all__ = ["build_graph", "run_agent", "AgentState", "NodeTrace"]
