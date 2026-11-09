import json

import pytest

from scripts.run_kg_pilot import main, prepare_pilot, select_chunks, load_active_chunks, lightrag_usage_auditable


def test_page_whitelist_excludes_rest_of_manual():
    rows = [{"source_doc": "manual", "page_start": p, "page_end": p,
             "chunk_id": f"ck_{p}", "content": str(p)} for p in range(1, 834)]
    selected = select_chunks(rows, {"manual": set(range(132, 152))})
    assert len(selected) == 20
    assert {r["page_start"] for r in selected} == set(range(132, 152))


def test_missing_or_cross_boundary_provenance_is_rejected():
    rows = [
        {"source_doc": "manual", "page_start": 132, "page_end": 133, "content": "ok"},
        {"source_doc": "manual", "page_start": 132, "page_end": 152, "content": "cross"},
        {"source_doc": "manual", "page_start": 132, "content": "missing end"},
        {"source_doc": "manual", "page_start": 132, "page_end": 10**9, "content": "absurd span"},
        {"source_doc": "other", "page_start": 132, "page_end": 132, "content": "wrong doc"},
    ]
    assert [r["content"] for r in select_chunks(rows, {"manual": set(range(132, 152))})] == ["ok"]


def test_prepare_isolated_manifest_and_source_markers(tmp_path):
    root = tmp_path / "lightrag_pilot"
    old = tmp_path / "lightrag"
    old.mkdir()
    (old / "sentinel").write_text("keep", encoding="utf-8")
    rows = [{"source_doc": "manual.pdf", "page_start": 132, "page_end": 132,
             "chunk_id": "ck_1", "content": "基金认购说明", "snapshot_id": "snap_active"}]
    batch = prepare_pilot(rows, {"manual.pdf": {132}}, root)
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    assert batch.is_relative_to(root)
    assert manifest["snapshot_id"] == "snap_active"
    assert not batch.is_relative_to(old)
    assert (old / "sentinel").read_text(encoding="utf-8") == "keep"
    assert manifest["chunks"] == [{"source_doc": "manual.pdf", "page_start": 132,
                                     "page_end": 132, "chunk_id": "ck_1",
                                     "file": "chunk_0001.txt"}]
    assert "manual.pdf#p132" in (batch / "chunk_0001.txt").read_text(encoding="utf-8")


def test_prepare_refuses_unscoped_or_oversized_input(tmp_path):
    root = tmp_path / "lightrag_pilot"
    with pytest.raises(ValueError):
        prepare_pilot([], {}, root)
    rows = [{"source_doc": "m", "page_start": 1, "page_end": 1,
             "content": "x" * 20_001, "chunk_id": "huge"}]
    with pytest.raises(ValueError):
        prepare_pilot(rows, {"m": {1}}, root)
    with pytest.raises(ValueError):
        prepare_pilot([{"source_doc": "m", "page_start": 1, "page_end": 1,
                        "content": "ok", "chunk_id": "ck"}], {"m": {1}},
                      tmp_path / "lightrag" / "lightrag_pilot")


def test_default_cli_is_dry_and_has_no_clear(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("scripts.run_kg_pilot.load_active_chunks", lambda allowlist: called.append("load") or [])
    assert main([]) == 0
    assert called == []
    with pytest.raises(SystemExit):
        main(["--clear"])
    assert lightrag_usage_auditable() is False
    assert "未发送" in capsys.readouterr().out


def test_loader_uses_manifest_sibling_chunks_and_rejects_stale_snapshot(tmp_path):
    active = tmp_path / "kb_full"
    active.mkdir()
    manifest = active / "kb_v3_manifest.json"
    manifest.write_text('{"snapshot_id":"snap_active","versions":[{"source_doc":"manual"}]}', encoding="utf-8")
    chunks = active / "chunks.jsonl"
    chunks.write_text(json.dumps({"source_doc":"manual", "page_start":1, "page_end":1,
                                  "content":"active", "snapshot_id":"snap_active"}) + "\n",
                      encoding="utf-8")
    assert load_active_chunks({"manual": {1}}, manifest)[0]["content"] == "active"
    chunks.write_text(json.dumps({"source_doc":"manual", "page_start":1, "page_end":1,
                                  "content":"stale", "snapshot_id":"snap_old"}) + "\n",
                      encoding="utf-8")
    with pytest.raises(ValueError, match="快照"):
        load_active_chunks({"manual": {1}}, manifest)
