"""Ark provider boundary: no live requests or real credentials."""

import httpx
import pytest

from src.contracts import ContractError, ErrorCode
from src.system.llm_api import DeepSeekAPIClient
from src.system.rerank import EvidenceReranker
from src.system.agent import GenerativeAnswerer


MODEL = "doubao-seed-2-1-pro-260915"


def _ark_client(monkeypatch, tmp_path, response):
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    monkeypatch.setenv("FINANCE_PILOT_BUDGET_PATH", str(tmp_path / "pilot.sqlite3"))
    client = DeepSeekAPIClient()
    seen = []

    def reply(request):
        seen.append(request)
        return response

    client.gateway.transport._client_factory = lambda: httpx.Client(
        transport=httpx.MockTransport(reply)
    )
    return client, seen


def test_ark_response_records_actual_usage_and_provider(monkeypatch, tmp_path):
    client, seen = _ark_client(monkeypatch, tmp_path, httpx.Response(200, json={
        "id": "ark-request-1",
        "model": MODEL,
        "choices": [{"message": {"content": "连接正常"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }))
    response = client.gateway.generate([{"role": "user", "content": "测试"}])
    assert client.gateway.snapshot.provider == "ark"
    assert response.provider == "ark"
    assert response.actual_model == MODEL
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (10, 2, 12)
    assert len(seen) == 1
    assert seen[0].url.path == "/api/v3/chat/completions"
    assert "fake-test-key" not in repr(client.gateway.transport)


def test_ark_chat_rejects_model_substitution(monkeypatch, tmp_path):
    client, seen = _ark_client(monkeypatch, tmp_path, httpx.Response(200, json={
        "model": "different-paid-model",
        "choices": [{"message": {"content": "错误模型"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }))
    with pytest.raises(ContractError) as error:
        client.chat([{"role": "user", "content": "测试"}])
    assert error.value.code == ErrorCode.MODEL_RESPONSE_INVALID
    assert len(seen) == 1


def test_ark_chat_rejects_missing_actual_model(monkeypatch, tmp_path):
    client, seen = _ark_client(monkeypatch, tmp_path, httpx.Response(200, json={
        "choices": [{"message": {"content": "不能确认模型"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }))
    with pytest.raises(ContractError) as error:
        client.chat([{"role": "user", "content": "测试"}])
    assert error.value.code == ErrorCode.MODEL_RESPONSE_INVALID
    assert len(seen) == 1


@pytest.mark.parametrize("status", [402, 429])
def test_ark_payment_and_rate_errors_do_not_retry(monkeypatch, tmp_path, status):
    client, seen = _ark_client(monkeypatch, tmp_path, httpx.Response(status, json={"error": "stop"}))
    with pytest.raises(ContractError) as error:
        client.chat([{"role": "user", "content": "测试"}])
    assert not error.value.retryable
    assert len(seen) == 1


def test_ark_key_enables_existing_generation_and_rerank_callers(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("FINANCE_API_KEY", raising=False)
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    monkeypatch.setenv("FINANCE_RERANK_BACKEND", "api")
    monkeypatch.setenv("FINANCE_GEN_BACKEND", "api")
    assert EvidenceReranker().api_client.provider == "ark"
    assert GenerativeAnswerer().api_client.provider == "ark"


def test_pilot_refuses_unapproved_ark_model_before_http(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
    monkeypatch.setenv("ARK_MODEL", "another-paid-model")
    with pytest.raises(ValueError, match="试跑模型"):
        DeepSeekAPIClient()


def test_pilot_refuses_non_official_ark_endpoint(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
    monkeypatch.setenv("ARK_MODEL", MODEL)
    monkeypatch.setenv("ARK_BASE_URL", "https://example.invalid/api/v3")
    with pytest.raises(ValueError, match="官方 API"):
        DeepSeekAPIClient()
