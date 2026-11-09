import base64
import hashlib
from pathlib import Path

import pytest

from src.agent.visual_answer import build_image_messages, pilot_visual_approved
from src.agent.nodes import AgentNodes
from src.agent.state import initial_state


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lXcAAAAASUVORK5CYII=")


def _evidence(path: Path) -> dict:
    return {"image_ref": str(path), "source_doc": "公开报告.pdf", "vision_page": 2,
            "image_sha256": hashlib.sha256(PNG).hexdigest(), "evidence_kind": "page_image"}


def test_reject_image_outside_root(tmp_path):
    allowed = tmp_path / "pages"
    allowed.mkdir()
    other = tmp_path / "private.png"
    other.write_bytes(PNG)
    with pytest.raises(ValueError, match="可信"):
        build_image_messages("图中是什么？", _evidence(other), allowed)


def test_reject_missing_oversized_or_identity_mismatch(tmp_path):
    allowed = tmp_path / "pages"
    allowed.mkdir()
    path = allowed / "page.png"
    with pytest.raises(ValueError):
        build_image_messages("图中是什么？", _evidence(path), allowed)
    path.write_bytes(PNG)
    with pytest.raises(ValueError):
        build_image_messages("图中是什么？", {**_evidence(path), "image_sha256": "bad"}, allowed)
    with pytest.raises(ValueError):
        build_image_messages("图中是什么？", {**_evidence(path), "image_sha256": None}, allowed)
    path.write_bytes(PNG + b"0" * 5_000_000)
    with pytest.raises(ValueError):
        build_image_messages("图中是什么？", _evidence(path), allowed)


def test_valid_page_has_text_and_one_image(tmp_path):
    allowed = tmp_path / "pages"
    allowed.mkdir()
    path = allowed / "page.png"
    path.write_bytes(PNG)
    messages = build_image_messages("图中是什么？", _evidence(path), allowed)
    parts = messages[0]["content"]
    assert len(messages) == 1 and len(parts) == 2
    assert "公开报告.pdf" in parts[0]["text"] and "第2页" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_visual_pilot_requires_exact_operator_approved_page_and_digest(monkeypatch):
    evidence = {"page_id": "vp_selected", "image_sha256": "a" * 64}
    monkeypatch.setenv("FINANCE_PILOT_VISUAL", "1")
    assert pilot_visual_approved(evidence) is False
    monkeypatch.setenv("FINANCE_PILOT_VISUAL_PAGE_ID", "vp_selected")
    monkeypatch.setenv("FINANCE_PILOT_VISUAL_SHA256", "a" * 64)
    assert pilot_visual_approved(evidence) is True
    assert pilot_visual_approved({**evidence, "page_id": "vp_other"}) is False
    assert pilot_visual_approved({**evidence, "image_sha256": "b" * 64}) is False


def test_mixed_visual_hit_can_reach_selected_vlm_without_other_page_upload(monkeypatch):
    calls = []
    class FakeLLM:
        enabled = True
        def chat(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "图中显示基金数据"

    monkeypatch.setenv("FINANCE_PILOT_VISUAL", "1")
    monkeypatch.setenv("FINANCE_PILOT_VISUAL_PAGE_ID", "vp_selected")
    monkeypatch.setenv("FINANCE_PILOT_VISUAL_SHA256", "a" * 64)
    monkeypatch.setattr("src.agent.visual_answer.build_image_messages",
                        lambda question, ev, root: [{"role": "user", "content": "selected image only"}])
    nodes = AgentNodes(retriever=object(), llm=FakeLLM(), reranker=None, log=lambda _: None)
    state = initial_state("图中是什么？")
    state["relevant_documents"] = [
        {"chunk_id": "text", "content": "图表标题", "source_doc": "报告.pdf"},
        {"block_id": "vp_selected", "page_id": "vp_selected", "image_sha256": "a" * 64,
         "image_ref": "selected.png", "evidence_kind": "page_image", "content": "",
         "source_doc": "报告.pdf", "vision_page": 1},
    ]
    result = nodes.generate(state)
    assert len(calls) == 1 and calls[0][1]["node"] == "visual_generate"
    assert "图像模型初步识别" in result["generation"]


def test_image_only_without_api_never_claims_visual_verification(tmp_path):
    nodes = AgentNodes(retriever=object(), llm=None, reranker=None, log=lambda _: None)
    state = initial_state("图中基金认购步骤是什么？")
    state["relevant_documents"] = [{"evidence_kind": "page_image", "source_doc": "图.pdf",
                                    "vision_page": 1, "content": "", "image_ref": "missing.png"}]
    result = nodes.generate(state)
    assert "未经图像内容验证" in result["generation"]
    assert "图中" not in result["generation"] or "步骤" not in result["generation"]


def test_image_reservation_is_fixed_not_base64_length(monkeypatch, tmp_path):
    from src.system import llm_api

    reserved = []
    class Budget:
        def __init__(self, *args, **kwargs):
            pass
        def reserve(self, _rid, tokens, **kwargs):
            reserved.append(tokens)
        def settle(self, *args):
            pass
    monkeypatch.setattr("src.system.pilot_budget.PilotBudget", Budget)
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "fake")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    client = llm_api.DeepSeekAPIClient()
    client.gateway.generate = lambda *args, **kwargs: type("R", (), {"text": "图像答案"})()
    client.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "x" * 100_000}}]}], max_tokens=500)
    assert reserved == [50_000]
