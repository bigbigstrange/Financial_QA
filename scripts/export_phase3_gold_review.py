#!/usr/bin/env python3
"""Export provisional gold evidence for human review."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="导出阶段 3 待人工审核证据")
    parser.add_argument(
        "--gold",
        default=str(ROOT / "output" / "phase3" / "golden_retrieval.json"),
    )
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "output" / "phase3" / "eval_corpus.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "phase3" / "gold_review.csv"),
    )
    args = parser.parse_args()

    corpus = {
        item["block_id"]: item
        for item in (
            json.loads(line)
            for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    golden = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "question_id",
                "question",
                "block_id",
                "source_doc",
                "page",
                "content_snippet",
                "review_status",
            ]
        )
        for case in golden["cases"]:
            for block_id in case["gold_block_ids"]:
                item = corpus.get(block_id) or {}
                content = str(item.get("content") or "")
                writer.writerow(
                    [
                        case["question_id"],
                        case["question"],
                        block_id,
                        item.get("source_doc") or "",
                        item.get("page") or "",
                        content[:300],
                        "pending_review",
                    ]
                )
    print(f"已导出待审核证据: {output}")


if __name__ == "__main__":
    main()
