import json
import sqlite3

from scripts.migrate_financial_eval_50 import migrate


def test_migration_proposes_new_snapshot_evidence_without_approving_it(tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"eval_id": 1, "question": "银行代理保险资料？",
                               "expected_source": "local", "should_refuse": False,
                               "answer_points": ["禁止误导销售"],
                               "acceptable_evidence": ["old-1"]},
                              ensure_ascii=False) + "\n", encoding="utf-8")
    db_path = tmp_path / "old.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE kb_fts(block_id TEXT, source_doc TEXT, content TEXT, page INTEGER)")
        db.execute("INSERT INTO kb_fts VALUES(?,?,?,?)",
                   ("old-1", "监管通知", "商业银行代理保险不得误导销售。", 3))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"snapshot_id":"new-snapshot"}', encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_id": "new-1", "document_version": "v1",
                                  "snapshot_id": "new-snapshot", "source_doc": "监管通知.docx",
                                  "page_start": 4, "content": "商业银行代理保险不得误导销售。"},
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "candidates.jsonl"
    summary = migrate(old, db_path, manifest, chunks, output)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["suggested_ids"] == 1
    assert row["review_status"] == "pending_snapshot_review"
    assert row["evidence_mappings"][0]["candidates"][0]["chunk_id"] == "new-1"
