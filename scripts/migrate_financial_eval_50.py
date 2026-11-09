"""Propose (never approve) mappings from the old 50-case set to the active KB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (ROOT.parent.parent / "Financial" / "finance-qa-platform" /
                  "finance-qa-platform")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _text(value: object) -> str:
    return re.sub(r"[^\w]+", "", str(value or "")).lower()


def _title(value: object) -> str:
    return _text(re.sub(r"\.(pdf|docx?|xlsx?|pptx?|txt|png|jpe?g)$", "",
                        str(value or ""), flags=re.IGNORECASE))


def _trigrams(value: str) -> set[str]:
    return {value[i:i + 3] for i in range(max(0, len(value) - 2))}


def _overlap(old_text: str, new_text: str) -> float:
    old, new = _text(old_text), _text(new_text)
    if len(old) < 12 or not new:
        return 0.0
    if old in new:
        return 1.0
    grams = _trigrams(old)
    return len(grams & _trigrams(new)) / len(grams) if grams else 0.0


def migrate(source_gold: Path, source_fts: Path, manifest: Path,
            chunks: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError("候选评测文件已存在，禁止覆盖")
    cases = [json.loads(line) for line in source_gold.open(encoding="utf-8") if line.strip()]
    old_ids = {str(item) for case in cases for item in case.get("acceptable_evidence") or []}
    old = {}
    uri = f"file:{source_fts.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        for block_id, title, content, page in db.execute(
                "SELECT block_id,source_doc,content,page FROM kb_fts"):
            if block_id in old_ids and block_id not in old:
                old[block_id] = {"source_doc": title, "content": content, "page": page}

    snapshot = json.loads(manifest.read_text(encoding="utf-8"))["snapshot_id"]
    current = defaultdict(list)
    with chunks.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("snapshot_id") == snapshot:
                current[_title(item.get("source_doc"))].append(item)

    mapped_ids = set()
    proposals = []
    for case in cases:
        mappings = []
        for old_id in case.get("acceptable_evidence") or []:
            legacy = old.get(str(old_id))
            candidates = current.get(_title(legacy["source_doc"])) if legacy else []
            ranked = sorted(((round(_overlap(legacy["content"], item.get("content")), 4), item)
                             for item in candidates), key=lambda pair: pair[0], reverse=True) if legacy else []
            suggestions = [{"chunk_id": item["chunk_id"],
                            "document_version": item["document_version"],
                            "source_doc": item["source_doc"],
                            "page_start": item.get("page_start"),
                            "text_overlap": score,
                            "excerpt": str(item.get("content") or "")[:360]}
                           for score, item in ranked[:2] if score >= 0.55]
            if suggestions:
                mapped_ids.add(str(old_id))
            mappings.append({"old_block_id": old_id,
                             "old_source_doc": legacy["source_doc"] if legacy else None,
                             "old_page": legacy["page"] if legacy else None,
                             "old_excerpt": str(legacy["content"])[:360] if legacy else None,
                             "candidates": suggestions})
        proposals.append({
            "case_id": f"financial50-{case['eval_id']:03d}",
            "question": case["question"],
            "expected_source": case.get("expected_source"),
            "expected_refusal": bool(case.get("should_refuse")),
            "answer_points_from_old_review": case.get("answer_points") or [],
            "source_snapshot": snapshot,
            "review_status": "pending_snapshot_review",
            "reviewer": "",
            "evidence_mappings": mappings,
            "note": "旧库块 ID 与新快照不同；候选仅按同名文档和文字重合建议，未经人工复核。",
        })

    summary = {"cases": len(cases), "old_evidence_ids": len(old_ids),
               "found_old_ids": len(old), "suggested_ids": len(mapped_ids),
               "cases_with_all_ids_suggested": sum(all(m["candidates"] for m in row["evidence_mappings"])
                                                  for row in proposals if row["evidence_mappings"]),
               "snapshot_id": snapshot, "source_gold_sha256": _sha256(source_gold),
               "manifest_sha256": _sha256(manifest), "chunks_sha256": _sha256(chunks),
               "status": "machine_candidates_not_gold"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in proposals:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gold", type=Path,
                        default=DEFAULT_SOURCE / "test" / "agent_eval_50.jsonl")
    parser.add_argument("--source-fts", type=Path, default=DEFAULT_SOURCE / "vector_cache" /
                        "agent_fts_kb_de1893cf442d68dd1de1ad7f.sqlite3")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "output" / "kb_full" / "kb_v3_manifest.json")
    parser.add_argument("--chunks", type=Path,
                        default=ROOT / "output" / "kb_full" / "chunks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "quality" /
                        "financial50-snapshot-candidates.jsonl")
    args = parser.parse_args()
    print(json.dumps(migrate(args.source_gold, args.source_fts, args.manifest,
                             args.chunks, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
