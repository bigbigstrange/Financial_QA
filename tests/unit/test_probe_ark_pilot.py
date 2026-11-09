"""One-shot pilot probe stays bounded and never prints a credential."""

import pytest

from src.system.llm_api import DeepSeekAPIClient, configured_api_key


class FakeClient:
    model = "doubao-seed-2-1-pro-260915"

    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "连接正常"


def test_probe_makes_one_small_call_without_printing_key(capsys):
    from scripts.probe_ark_pilot import probe

    client = FakeClient()
    assert probe(client) == "连接正常"
    assert len(client.calls) == 1
    assert client.calls[0][1]["max_tokens"] == 16
    assert client.calls[0][1]["node"] == "probe"
    assert "fake-test-key" not in capsys.readouterr().out


def test_key_file_is_loaded_without_copying_it_into_env(tmp_path, monkeypatch):
    path = tmp_path / "key.txt"
    path.write_text("ark-fake-test-key\n", encoding="utf-8")
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY_FILE", str(path))
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    assert configured_api_key() == "ark-fake-test-key"


def test_multiline_key_file_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "key.txt"
    path.write_text("ark-first\nark-second\n", encoding="utf-8")
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY_FILE", str(path))
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="单行"):
        configured_api_key()


def test_paid_disable_blocks_probe_before_network(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "ark-fake-test-key")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "1")
    client = DeepSeekAPIClient()
    with pytest.raises(RuntimeError, match="已暂停"):
        client.chat([{"role": "user", "content": "测试"}], max_tokens=16)
