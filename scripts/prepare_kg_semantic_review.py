"""Freeze an evidence-bound, cross-document KG sample for semantic review."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare(selection: Path, checkpoint: Path, output: Path, *, count: int = 30,
            seed: int = 20260924) -> dict:
    if output.exists():
        raise FileExistsError(f"语义审核文件已存在，禁止覆盖：{output}")
    if count < 1:
        raise ValueError("count 必须大于 0")
    frozen = json.loads(selection.read_text(encoding="utf-8"))
    allowed = {chunk for ids in frozen["chunk_ids"].values() for chunk in ids}
    with sqlite3.connect(f"file:{checkpoint.as_posix()}?mode=ro", uri=True) as db:
        rows = db.execute("""SELECT snapshot_id,chunk_id,document_version,source_doc,page,triples_json
                             FROM chunks WHERE status='completed'""").fetchall()
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for snapshot, chunk_id, version, source, page, payload in rows:
        if snapshot != frozen["snapshot_id"] or chunk_id not in allowed:
            continue
        for index, triple in enumerate(json.loads(payload)):
            evidence = str(triple.get("evidence") or "")
            if not evidence or any(str(triple.get(key) or "") not in evidence
                                   for key in ("head", "relation", "tail")):
                continue
            by_doc[version].append({
                "snapshot_id": snapshot, "chunk_id": chunk_id,
                "document_version": version, "source_doc": source,
                "page": page if page > 0 else None,
                "triple_index": index, "head": triple["head"],
                "relation": triple["relation"], "tail": triple["tail"],
                "evidence": evidence, "review_status": "pending_review",
                "semantic_valid": None, "reviewer": "", "review_note": "",
            })
    rng = random.Random(seed)
    versions = sorted(by_doc)
    rng.shuffle(versions)
    sample = []
    for version in versions:
        sample.append(rng.choice(by_doc[version]))
        if len(sample) >= count:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in sample:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"sampled_relations": len(sample), "distinct_documents": len(sample),
            "review_status": "pending_review", "source_snapshot": frozen["snapshot_id"],
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "output" / "kg_pilot" / "selection_300.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "output" / "kg_pilot" / "checkpoint.sqlite3")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "kg_pilot" / "semantic_review_30.jsonl")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.checkpoint, args.output,
                             count=args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
