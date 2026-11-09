import json

import pytest

from src.agent import build_graph, run_agent
from src.system.execution_control import execution_guard


def test_cancel_between_nodes_blocks_further_model_calls():
    calls = []

    class LLM:
        enabled, provider, model = True, "ark", "fixed"

        def chat(self, _messages, **kwargs):
            calls.append(kwargs["node"])
            return json.dumps({"route": "direct"})

    def guard():
        if calls:
            raise RuntimeError("cancelled by owner")

    token = execution_guard.set(guard)
    try:
        with pytest.raises(RuntimeError, match="cancelled by owner"):
            run_agent(build_graph(object(), LLM()), "你好")
        assert calls == ["route_query"]
    finally:
        execution_guard.reset(token)


def test_deadline_blocks_call_before_budget_reservation(monkeypatch):
    from src.system.llm_api import DeepSeekAPIClient
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "deepseek")
    client = DeepSeekAPIClient(api_key="test-not-used")
    token = execution_guard.set(lambda: (_ for _ in ()).throw(TimeoutError("deadline")))
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            client.chat([])
        assert client.last_response is None
    finally:
        execution_guard.reset(token)
