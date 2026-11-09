"""Evidence-bound Ark graph pilot over the active local text snapshot.

This is intentionally a pilot, not a claim that every document has a graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

from src.system.llm_api import DeepSeekAPIClient
from src.system.pilot_budget import PilotBudget, PilotBudgetStopped


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "output" / "kg_pilot" / "checkpoint.sqlite3"
GENERIC_ENTITIES = {"我们", "本报告", "本公司", "公司", "客户", "该行", "银行", "图1", "图2"}


class KGCheckpoint:
    """Never reissue an uncertain paid request after process interruption."""

    def __init__(self, path: Path, *, recover_interrupted: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS chunks (
                snapshot_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                document_version TEXT NOT NULL, source_doc TEXT NOT NULL,
                page INTEGER NOT NULL, content_sha256 TEXT NOT NULL,
                status TEXT NOT NULL, triples_json TEXT NOT NULL DEFAULT '[]',
                total_tokens INTEGER NOT NULL DEFAULT 0, error_code TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(snapshot_id, chunk_id)
            )""")
            if recover_interrupted:
                db.execute("UPDATE chunks SET status='needs_review' WHERE status='started'")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _identity(row):
        return row["snapshot_id"], row["chunk_id"]

    def start(self, row: dict) -> bool:
        with self._connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO chunks
                (snapshot_id,chunk_id,document_version,source_doc,page,content_sha256,status)
                VALUES(?,?,?,?,?,?,'started')""", (
                row["snapshot_id"], row["chunk_id"], row["document_version"],
                row["source_doc"], int(row["page"] or 0),  # 0 = unpaginated text, never a cited page
                hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
            ))
            return cursor.rowcount == 1

    def complete(self, row: dict, triples: list[dict], total_tokens: int = 0) -> bool:
        status = "completed" if triples else "no_relations"
        with self._connect() as db:
            cursor = db.execute("""UPDATE chunks SET status=?,triples_json=?,total_tokens=?,
                updated_at=CURRENT_TIMESTAMP WHERE snapshot_id=? AND chunk_id=? AND status='started'""", (
                status, json.dumps(triples, ensure_ascii=False), int(total_tokens), *self._identity(row),
            ))
            return cursor.rowcount == 1

    def fail(self, row: dict, status: str, code: str) -> None:
        if status not in {"needs_review", "invalid_response", "budget_stopped"}:
            raise ValueError("invalid failure status")
        with self._connect() as db:
            db.execute("""UPDATE chunks SET status=?,error_code=?,updated_at=CURRENT_TIMESTAMP
                WHERE snapshot_id=? AND chunk_id=? AND status='started'""",
                (status, code[:64], *self._identity(row)))

    def summary(self) -> dict[str, int]:
        with self._connect() as db:
            return dict(Counter({status: count for status, count in db.execute(
                "SELECT status,COUNT(*) FROM chunks GROUP BY status")}))

    def completed_triples(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT triples_json FROM chunks WHERE status='completed'").fetchall()
        return [triple for (payload,) in rows for triple in json.loads(payload)]

    def statuses(self, snapshot_id: str) -> dict[str, str]:
        with self._connect() as db:
            return dict(db.execute(
                "SELECT chunk_id,status FROM chunks WHERE snapshot_id=?", (snapshot_id,)))


def select_documents(rows: list[dict], priority: set[str], *, limit: int,
                     priority_limit: int) -> dict[str, list[dict]]:
    """Select complete, shortest text documents; keep candidate evidence represented."""
    if limit < 1 or priority_limit < 0 or priority_limit > limit:
        raise ValueError("invalid_document_limits")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["document_version"], []).append(row)
    ranked = sorted(groups, key=lambda doc: (len(groups[doc]), doc))
    priority_docs = {row["document_version"] for row in rows if row["chunk_id"] in priority}
    chosen = [doc for doc in ranked if doc in priority_docs][:priority_limit]
    chosen += [doc for doc in ranked if doc not in priority_docs][:limit - len(chosen)]
    if len(chosen) != limit:
        raise ValueError("insufficient_documents")
    return {doc: sorted(groups[doc], key=lambda row: row["chunk_id"]) for doc in chosen}


def freeze_selection(path: Path, snapshot_id: str, groups: dict[str, list[dict]]) -> None:
    payload = {"snapshot_id": snapshot_id, "documents": list(groups),
               "chunk_ids": {doc: [row["chunk_id"] for row in rows]
                             for doc, rows in groups.items()}}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("selection_changed: frozen document selection differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def document_progress(checkpoint: KGCheckpoint, groups: dict[str, list[dict]]) -> dict[str, int]:
    snapshot_id = next(iter(groups.values()))[0]["snapshot_id"] if groups else ""
    statuses = checkpoint.statuses(snapshot_id)
    complete = partial = pending = 0
    for rows in groups.values():
        states = [statuses.get(row["chunk_id"]) for row in rows]
        if all(state in {"completed", "no_relations"} for state in states):
            complete += 1
        elif any(states):
            partial += 1
        else:
            pending += 1
    return {"complete_documents": complete, "partial_documents": partial,
            "pending_documents": pending, "selected_documents": len(groups)}


def extract_chunk(row: dict, llm: DeepSeekAPIClient) -> list[dict]:
    content = row["content"].strip()
    messages = [
        {"role": "system", "content": (
            "从给定金融文档片段抽取最多 5 条明确的实体关系。只返回 JSON："
            '{"triples":[{"head":"","relation":"","tail":"","evidence":""}]}。'
            "evidence 必须逐字复制原文中的连续片段；不能推测或补充事实。没有明确关系返回空数组。"
        )},
        {"role": "user", "content": content},
    ]
    raw = llm.chat(messages, temperature=0, max_tokens=700, node=f"kg:{row['chunk_id']}")
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, re.I | re.S)
    payload = json.loads(match.group(1) if match else raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("triples"), list):
        raise ValueError("invalid_triples_schema")
    result = []
    for item in payload["triples"][:5]:
        if not isinstance(item, dict):
            continue
        fields = {key: str(item.get(key, "")).strip() for key in
                  ("head", "relation", "tail", "evidence")}
        if not all(fields.values()) or len(fields["evidence"]) < 6:
            continue
        if fields["evidence"] not in content:
            continue
        if (fields["head"] not in fields["evidence"] or
                fields["tail"] not in fields["evidence"] or
                fields["relation"] not in fields["evidence"]):
            continue
        if (fields["head"] in GENERIC_ENTITIES or fields["tail"] in GENERIC_ENTITIES or
                fields["relation"] in {"标题为", "为"}):
            continue
        result.append({**fields, **{key: row[key] for key in (
            "snapshot_id", "chunk_id", "document_version", "source_doc", "page")}})
    return result


def _ordered_chunks(root: Path, snapshot_id: str):
    candidates_path = root / "output" / "eval" / "gold_evidence_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.exists() else {}
    priority = set()
    def collect(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "chunk_id" and isinstance(nested, str):
                    priority.add(nested)
                else:
                    collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
    collect(candidates)
    rows = []
    with (root / "output" / "kb_full" / "chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("snapshot_id") == snapshot_id and row.get("content", "").strip():
                rows.append(row)
    rows.sort(key=lambda row: (row["chunk_id"] not in priority, row["chunk_id"]))
    return rows, priority


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chunks", type=int, default=1,
                        help="New paid chunks this invocation; 0 means until budget stops")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DB)
    parser.add_argument("--document-limit", type=int, default=0,
                        help="Fixed number of complete shortest documents; 0 keeps original chunk order")
    parser.add_argument("--priority-documents", type=int, default=20)
    parser.add_argument("--max-documents", type=int, default=0,
                        help="New complete documents this run; 0 means no document cap")
    parser.add_argument("--selection-file", type=Path,
                        default=ROOT / "output" / "kg_pilot" / "selection_300.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_chunks < 0:
        parser.error("--max-chunks must be >= 0")
    if args.document_limit < 0 or args.max_documents < 0:
        parser.error("document limits must be >= 0")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    manifest = json.loads((ROOT / "output" / "kb_full" / "kb_v3_manifest.json").read_text(encoding="utf-8"))
    rows, priority = _ordered_chunks(ROOT, manifest["snapshot_id"])
    groups = (select_documents(rows, priority, limit=args.document_limit,
                               priority_limit=args.priority_documents)
              if args.document_limit else {})
    if args.dry_run:
        print(json.dumps({"snapshot_id": manifest["snapshot_id"], "eligible_chunks": len(rows),
                          "priority_ids": len(priority), "selected_documents": len(groups),
                          "selected_chunks": sum(map(len, groups.values()))}, ensure_ascii=False))
        return 0
    if groups:
        freeze_selection(args.selection_file, manifest["snapshot_id"], groups)
    if (os.environ.get("FINANCE_LLM_PROVIDER") or "").lower() != "ark":
        raise RuntimeError("KG pilot requires the approved Ark provider")
    llm = DeepSeekAPIClient()
    if not llm.enabled:
        raise RuntimeError("Approved Ark API is not enabled")
    checkpoint = KGCheckpoint(args.checkpoint)
    budget = PilotBudget(Path(os.environ.get("FINANCE_PILOT_BUDGET_PATH") or
                              ROOT / "output" / "pilot_budget.sqlite3"),
                         limit=int(os.environ.get("FINANCE_PILOT_TOKEN_LIMIT") or 1_500_000),
                         model=llm.model)
    attempted = new_documents = invalid_count = 0
    stopped = False
    work = list(groups.values()) if groups else [rows]
    for doc_rows in work:
        if args.max_documents and new_documents >= args.max_documents:
            break
        before = document_progress(checkpoint, {doc_rows[0]["document_version"]: doc_rows}) if groups else None
        for row in doc_rows:
            if args.max_chunks and attempted >= args.max_chunks:
                stopped = True
                break
            if not checkpoint.start(row):
                continue
            attempted += 1
            try:
                triples = extract_chunk(row, llm)
                tokens = int(llm.last_response.total_tokens) if llm.last_response else 0
                checkpoint.complete(row, triples, tokens)
            except PilotBudgetStopped:
                reason = budget.summary()["stop_reason"] or "not_reserved"
                checkpoint.fail(row, "budget_stopped" if reason == "budget_exhausted" else "needs_review", reason)
                stopped = True
                break
            except (ValueError, json.JSONDecodeError):
                checkpoint.fail(row, "invalid_response", "invalid_json_or_evidence")
                invalid_count += 1
                if invalid_count >= 3:
                    stopped = True
                    break
                # The response has already settled; isolate this chunk, never retry it.
                continue
            except Exception:
                checkpoint.fail(row, "needs_review", "unexpected_call_failure")
                stopped = True
                break
        if groups and before["complete_documents"] == 0:
            after = document_progress(checkpoint, {doc_rows[0]["document_version"]: doc_rows})
            new_documents += after["complete_documents"]
        if stopped:
            break
    print(json.dumps({"snapshot_id": manifest["snapshot_id"], "attempted_this_run": attempted,
                      "checkpoint": checkpoint.summary(), "budget": budget.summary(),
                      "documents": document_progress(checkpoint, groups) if groups else None},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
