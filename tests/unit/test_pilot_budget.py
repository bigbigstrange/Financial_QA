"""Fail-closed token accounting; all HTTP is mocked."""

import httpx
import pytest
import sqlite3
from dataclasses import replace

from src.models.gateway import ModelResponse
from src.system.llm_api import DeepSeekAPIClient


MODEL = "doubao-seed-2-1-pro-260915"


def _budget_types():
    from src.system.pilot_budget import PilotBudget, PilotBudgetStopped
    return PilotBudget, PilotBudgetStopped


def _response(*, model=MODEL, total=12):
    return ModelResponse(
        text="ok", requested_model=MODEL, actual_model=model,
        provider="ark", request_id="r1", prompt_tokens=10,
        completion_tokens=2, total_tokens=total,
    )


def test_two_budget_objects_cannot_overreserve(tmp_path):
    PilotBudget, PilotBudgetStopped = _budget_types()
    path = tmp_path / "pilot.sqlite3"
    first, second = PilotBudget(path), PilotBudget(path)
    first.reserve("a", 900_000)
    with pytest.raises(PilotBudgetStopped):
        second.reserve("b", 600_001)
    assert second.summary()["reserved_tokens"] == 900_000
    with pytest.raises(PilotBudgetStopped):
        second.reserve("too-large", 1_500_001)


def test_only_one_paid_call_can_be_in_flight(tmp_path):
    PilotBudget, PilotBudgetStopped = _budget_types()
    first = PilotBudget(tmp_path / "pilot.sqlite3")
    second = PilotBudget(tmp_path / "pilot.sqlite3")
    first.reserve("a", 100)
    with pytest.raises(PilotBudgetStopped):
        second.reserve("b", 100)
    first.settle("a", _response(), "route_query")
    second.reserve("b", 100)
    assert second.summary()["reserved_tokens"] == 100


def test_pilot_limit_cannot_be_raised_above_approved_ceiling(tmp_path):
    PilotBudget, _ = _budget_types()
    assert PilotBudget(tmp_path / "pilot.sqlite3", limit=2_000_000).limit == 1_500_000
    with pytest.raises(ValueError):
        PilotBudget(tmp_path / "other.sqlite3", limit=0)


def test_inconsistent_usage_stops_future_paid_calls(tmp_path):
    PilotBudget, PilotBudgetStopped = _budget_types()
    budget = PilotBudget(tmp_path / "pilot.sqlite3")
    budget.reserve("a", 100)
    with pytest.raises(PilotBudgetStopped):
        budget.settle("a", _response(total=1), "generate")
    assert budget.summary()["stopped"] is True
    with pytest.raises(PilotBudgetStopped):
        budget.reserve("b", 100)


def test_ledger_audits_node_and_latency_even_on_failed_call(tmp_path):
    PilotBudget, _ = _budget_types()
    path = tmp_path / "pilot.sqlite3"
    budget = PilotBudget(path)
    budget.reserve("ok", 100, node="generate")
    budget.settle("ok", replace(_response(), latency_ms=17.5), "generate")
    budget.reserve("failed", 100, node="check_answer")
    budget.stop("failed", "http_402", latency_ms=25.0)
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT node,latency_ms,status FROM calls ORDER BY rowid").fetchall()
    assert rows == [("generate", 17.5, "settled"), ("check_answer", 25.0, "uncertain")]


def test_settle_records_actual_usage_without_prompt_or_secret(tmp_path):
    PilotBudget, _ = _budget_types()
    path = tmp_path / "pilot.sqlite3"
    budget = PilotBudget(path)
    budget.reserve("a", 100)
    budget.settle("a", _response(), "route_query")
    assert budget.summary()["spent_tokens"] == 12
    assert budget.summary()["reserved_tokens"] == 0
    content = path.read_bytes()
    assert b"fake-test-key" not in content
    assert "原文内容".encode("utf-8") not in content


@pytest.mark.parametrize("response", [_response(total=None), _response(model="other-model")])
def test_missing_usage_or_model_mismatch_stops_future_calls(tmp_path, response):
    PilotBudget, PilotBudgetStopped = _budget_types()
    budget = PilotBudget(tmp_path / "pilot.sqlite3")
    budget.reserve("a", 100)
    with pytest.raises(PilotBudgetStopped):
        budget.settle("a", response, "generate")
    with pytest.raises(PilotBudgetStopped):
        budget.reserve("b", 100)
    assert budget.summary()["stopped"] is True


@pytest.mark.parametrize("status", [402, 429])
def test_client_stops_after_payment_or_rate_error(tmp_path, monkeypatch, status):
    _, PilotBudgetStopped = _budget_types()
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    monkeypatch.setenv("FINANCE_PILOT_BUDGET_PATH", str(tmp_path / "pilot.sqlite3"))
    client = DeepSeekAPIClient()
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "stop"})

    client.gateway.transport._client_factory = lambda: httpx.Client(
        transport=httpx.MockTransport(reply)
    )
    with pytest.raises(PilotBudgetStopped):
        client.chat([{"role": "user", "content": "测试"}], max_tokens=16)
    with pytest.raises(PilotBudgetStopped):
        client.chat([{"role": "user", "content": "不能再调用"}], max_tokens=16)
    assert len(calls) == 1


def test_ark_client_request_header_matches_budget_ledger(tmp_path, monkeypatch):
    path = tmp_path / "pilot.sqlite3"
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    monkeypatch.setenv("FINANCE_PILOT_BUDGET_PATH", str(path))
    client = DeepSeekAPIClient()
    client_ids = []

    def reply(request):
        client_ids.append(request.headers.get("X-Client-Request-Id"))
        return httpx.Response(200, json={
            "id": "remote-id", "model": MODEL,
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                      "total_tokens": 12},
        })

    client.gateway.transport._client_factory = lambda: httpx.Client(
        transport=httpx.MockTransport(reply)
    )
    assert client.chat([{"role": "user", "content": "测试"}], max_tokens=16) == "ok"
    with sqlite3.connect(path) as db:
        ledger_id, status = db.execute(
            "SELECT request_id,status FROM calls"
        ).fetchone()
    assert client_ids == [ledger_id]
    assert status == "settled"


def test_uncertain_call_keeps_reservation_after_manual_resume(tmp_path):
    PilotBudget, PilotBudgetStopped = _budget_types()
    path = tmp_path / "pilot.sqlite3"
    budget = PilotBudget(path, limit=100)
    budget.reserve("uncertain", 70)
    budget.stop("uncertain", "request_uncertain")
    assert budget.summary()["uncertain_reserved_tokens"] == 70
    assert budget.summary()["reserved_tokens"] == 0
    with sqlite3.connect(path) as db:
        db.execute("UPDATE state SET stopped_reason=NULL WHERE id=1")
    budget.reserve("new", 30)
    budget.settle("new", _response(total=30), "chat")
    with pytest.raises(PilotBudgetStopped):
        budget.reserve("would-overrun", 1)
    assert budget.summary()["stop_reason"] == "budget_exhausted"
