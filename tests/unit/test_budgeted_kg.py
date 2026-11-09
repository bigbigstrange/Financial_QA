"""Paid graph extraction is evidence-bound and resumable without duplicate calls."""

import json
import sys
import sqlite3
from pathlib import Path

from scripts import build_budgeted_kg as kg
from scripts.build_budgeted_kg import (KGCheckpoint, extract_chunk, select_documents,
                                       document_progress, freeze_selection)
from src.system.kg_pilot_channel import KGPilotRetriever


ROW = {
    "snapshot_id": "snap_a",
    "chunk_id": "chunk_1",
    "document_version": "dv_1",
    "source_doc": "bank.pdf",
    "page": 3,
    "content": "甲银行提供基金认购服务，客户需先完成风险测评。",
}


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self.text


def test_extract_accepts_only_verbatim_evidence():
    llm = FakeLLM(json.dumps({"triples": [
        {"head": "甲银行", "relation": "提供", "tail": "基金认购服务",
         "evidence": "甲银行提供基金认购服务"},
        {"head": "乙银行", "relation": "提供", "tail": "贷款",
         "evidence": "乙银行提供贷款"},
    ]}, ensure_ascii=False))

    triples = extract_chunk(ROW, llm)

    assert len(triples) == 1
    assert triples[0]["chunk_id"] == "chunk_1"
    assert triples[0]["page"] == 3
    assert triples[0]["evidence"] == "甲银行提供基金认购服务"
    assert llm.calls == 1
    assert llm.last_kwargs["max_tokens"] == 700


def test_extract_rejects_generic_subject_and_heading_relation():
    llm = FakeLLM(json.dumps({"triples": [
        {"head": "我们", "relation": "提供", "tail": "基金认购服务",
         "evidence": "我们提供基金认购服务"},
        {"head": "甲银行", "relation": "标题为", "tail": "基金认购服务",
         "evidence": "甲银行提供基金认购服务"},
    ]}, ensure_ascii=False))
    row = {**ROW, "content": ROW["content"] + "我们提供基金认购服务"}
    assert extract_chunk(row, llm) == []


def test_checkpoint_never_retries_started_or_completed_chunk(tmp_path):
    path = tmp_path / "kg.sqlite3"
    checkpoint = KGCheckpoint(path)
    assert checkpoint.start(ROW) is True
    assert checkpoint.start(ROW) is False

    recovered = KGCheckpoint(path)
    assert recovered.start(ROW) is False
    assert recovered.summary()["needs_review"] == 1
    assert recovered.complete(ROW, []) is False


def test_completed_chunk_is_skipped_and_readable(tmp_path):
    path = tmp_path / "kg.sqlite3"
    checkpoint = KGCheckpoint(path)
    assert checkpoint.start(ROW) is True
    triple = {"head": "甲银行", "relation": "提供", "tail": "基金认购服务",
              "evidence": "甲银行提供基金认购服务", "chunk_id": "chunk_1",
              "snapshot_id": "snap_a", "document_version": "dv_1",
              "source_doc": "bank.pdf", "page": 3}
    assert checkpoint.complete(ROW, [triple], 42) is True

    resumed = KGCheckpoint(path)
    assert resumed.start(ROW) is False
    assert resumed.summary()["completed"] == 1
    assert resumed.completed_triples() == [triple]


def test_unpaginated_text_chunk_uses_internal_zero_sentinel(tmp_path):
    row = {**ROW, "page": None, "source_doc": "notes.txt"}
    checkpoint = KGCheckpoint(tmp_path / "kg.sqlite3")
    assert checkpoint.start(row)
    triples = extract_chunk(row, FakeLLM(json.dumps({"triples": [{
        "head": "甲银行", "relation": "提供", "tail": "基金认购服务",
        "evidence": "甲银行提供基金认购服务",
    }]}, ensure_ascii=False)))
    assert checkpoint.complete(row, triples, 40)
    with sqlite3.connect(checkpoint.path) as db:
        assert db.execute("SELECT page FROM chunks").fetchone() == (0,)
    assert checkpoint.completed_triples()[0]["page"] is None


def test_graph_channel_maps_only_verified_active_chunk(tmp_path):
    checkpoint = KGCheckpoint(tmp_path / "kg.sqlite3")
    assert checkpoint.start(ROW)
    triples = extract_chunk(ROW, FakeLLM(json.dumps({"triples": [{
        "head": "甲银行", "relation": "提供", "tail": "基金认购服务",
        "evidence": "甲银行提供基金认购服务",
    }]}, ensure_ascii=False)))
    assert checkpoint.complete(ROW, triples, 40)
    blocks = [
        {**ROW, "block_id": "chunk_1"},
        {**ROW, "chunk_id": "chunk_2", "block_id": "chunk_2",
         "document_version": "dv_other", "source_doc": "乙银行.pdf",
         "content": "乙银行提供基金认购服务"},
    ]
    channel = KGPilotRetriever(blocks, checkpoint)
    hits = channel.search("甲银行的基金认购服务", top_k=5)
    assert [hit["block_id"] for hit in hits] == ["chunk_1"]
    assert hits[0]["document_version"] == "dv_1"
    assert hits[0]["kg_evidence"] == "甲银行提供基金认购服务"
    assert channel.search("甲银行有哪些基金认购服务", top_k=5)
    assert channel.search("乙银行的贷款", top_k=5) == []


def test_graph_relation_matches_natural_numeric_question(tmp_path):
    row = {**ROW, "content": "甲银行2023年末不良率为1.36%。"}
    checkpoint = KGCheckpoint(tmp_path / "kg.sqlite3")
    assert checkpoint.start(row)
    triples = extract_chunk(row, FakeLLM(json.dumps({"triples": [{
        "head": "甲银行", "relation": "不良率为", "tail": "1.36%",
        "evidence": "甲银行2023年末不良率为1.36%",
    }]}, ensure_ascii=False)))
    assert checkpoint.complete(row, triples, 40)
    channel = KGPilotRetriever([{**row, "block_id": "chunk_1"}], checkpoint)
    assert channel.search("甲银行2023年末不良率是多少")


def test_graph_channel_rejects_legacy_paraphrased_relation():
    row = {**ROW, "content": "甲银行证券代码：601658.SH"}
    triple = {"head": "甲银行", "relation": "证券代码为", "tail": "601658.SH",
              "evidence": row["content"], **{key: row[key] for key in
              ("snapshot_id", "chunk_id", "document_version", "source_doc", "page")}}

    class LegacyCheckpoint:
        def completed_triples(self):
            return [triple]

    channel = KGPilotRetriever([{**row, "block_id": row["chunk_id"]}], LegacyCheckpoint())
    assert channel.search("甲银行证券代码为多少") == []


def test_selects_whole_short_documents_with_priority():
    rows = [{**ROW, "chunk_id": f"{doc}_{number}", "document_version": doc}
            for doc, size in (("priority_long", 3), ("priority_short", 1),
                              ("other_short", 1), ("other_medium", 2))
            for number in range(size)]
    selected = select_documents(rows, {"priority_long_0", "priority_short_0"},
                                limit=3, priority_limit=1)
    assert list(selected) == ["priority_short", "other_short", "other_medium"]
    assert [len(group) for group in selected.values()] == [1, 1, 2]


def test_document_progress_requires_every_chunk_settled(tmp_path):
    one = {**ROW, "chunk_id": "one", "document_version": "doc_a"}
    two = {**ROW, "chunk_id": "two", "document_version": "doc_a"}
    three = {**ROW, "chunk_id": "three", "document_version": "doc_b"}
    checkpoint = KGCheckpoint(tmp_path / "kg.sqlite3")
    assert checkpoint.start(one)
    assert checkpoint.complete(one, [])
    assert checkpoint.start(two)
    checkpoint.fail(two, "needs_review", "request_uncertain")
    assert checkpoint.start(three)
    assert checkpoint.complete(three, [])
    progress = document_progress(checkpoint, {"doc_a": [one, two], "doc_b": [three]})
    assert progress == {"complete_documents": 1, "partial_documents": 1,
                        "pending_documents": 0, "selected_documents": 2}


def test_selection_freezes_snapshot_and_chunk_membership(tmp_path):
    groups = {"doc_a": [{**ROW, "chunk_id": "one"}]}
    path = tmp_path / "selection.json"
    freeze_selection(path, "snap_a", groups)
    freeze_selection(path, "snap_a", groups)
    assert json.loads(path.read_text(encoding="utf-8"))["documents"] == ["doc_a"]
    import pytest
    with pytest.raises(ValueError, match="selection_changed"):
        freeze_selection(path, "snap_b", groups)
    with pytest.raises(ValueError, match="selection_changed"):
        freeze_selection(path, "snap_a", {"doc_a": [{**ROW, "chunk_id": "changed"}]})


def test_invalid_response_is_isolated_without_retrying_its_chunk(tmp_path, monkeypatch):
    manifest = tmp_path / "output" / "kb_full" / "kb_v3_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"snapshot_id": "snap_a"}), encoding="utf-8")
    rows = [{**ROW, "document_version": f"doc_{i}", "chunk_id": f"chunk_{i}"}
            for i in range(2)]
    monkeypatch.setattr(kg, "ROOT", tmp_path)
    monkeypatch.setattr(kg, "_ordered_chunks", lambda *_: (rows, set()))
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")

    class LLM:
        enabled = True
        model = "test-model"
        last_response = None

        def chat(self, messages, **kwargs):
            self.last_response = type("Response", (), {"total_tokens": 42})()
            return "not json" if kwargs["node"] == "kg:chunk_0" else '{"triples":[]}'

    class Budget:
        def __init__(self, *args, **kwargs):
            pass

        def summary(self):
            return {"stopped": False, "stop_reason": None}

    monkeypatch.setattr(kg, "DeepSeekAPIClient", LLM)
    monkeypatch.setattr(kg, "PilotBudget", Budget)
    monkeypatch.setattr(sys, "argv", ["kg", "--document-limit", "2", "--priority-documents", "0",
                                    "--max-chunks", "0", "--checkpoint", str(tmp_path / "kg.sqlite3"),
                                    "--selection-file", str(tmp_path / "selection.json")])
    assert kg.main() == 0
    statuses = KGCheckpoint(tmp_path / "kg.sqlite3").statuses("snap_a")
    assert statuses == {"chunk_0": "invalid_response", "chunk_1": "no_relations"}


def test_three_invalid_responses_stop_the_batch(tmp_path, monkeypatch):
    manifest = tmp_path / "output" / "kb_full" / "kb_v3_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"snapshot_id": "snap_a"}), encoding="utf-8")
    rows = [{**ROW, "document_version": f"doc_{i}", "chunk_id": f"chunk_{i}"}
            for i in range(4)]
    monkeypatch.setattr(kg, "ROOT", tmp_path)
    monkeypatch.setattr(kg, "_ordered_chunks", lambda *_: (rows, set()))
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")

    class LLM:
        enabled = True
        model = "test-model"
        last_response = None

        def chat(self, messages, **kwargs):
            return "not json"

    class Budget:
        def __init__(self, *args, **kwargs):
            pass

        def summary(self):
            return {"stopped": False, "stop_reason": None}

    monkeypatch.setattr(kg, "DeepSeekAPIClient", LLM)
    monkeypatch.setattr(kg, "PilotBudget", Budget)
    monkeypatch.setattr(sys, "argv", ["kg", "--document-limit", "4", "--priority-documents", "0",
                                    "--max-chunks", "0", "--checkpoint", str(tmp_path / "kg.sqlite3"),
                                    "--selection-file", str(tmp_path / "selection.json")])
    kg.main()
    statuses = KGCheckpoint(tmp_path / "kg.sqlite3").statuses("snap_a")
    assert statuses == {f"chunk_{i}": "invalid_response" for i in range(3)}
