#!/usr/bin/env python3
"""Build a small standard-evidence retrieval corpus from the legacy KB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_gold(baseline_report: Path) -> tuple[dict[str, dict], int]:
    data = json.loads(baseline_report.read_text(encoding="utf-8"))
    gold: dict[str, dict] = {}
    count = 0
    for case in data.get("cases") or []:
        qid = str(case["question_id"])
        ids = [item.get("block_id") for item in case.get("evidence") or []]
        ids = [item for item in ids if item]
        gold[qid] = {
            "question": str(case.get("question") or ""),
            "gold_block_ids": ids,
        }
        count += len(ids)
    return gold, count


def build_corpus(
    kb_path: Path,
    gold: dict[str, list[str]],
    *,
    background_target: int,
) -> list[dict]:
    import ijson

    wanted = {block_id for entry in gold.values() for block_id in entry["gold_block_ids"]}
    collected: dict[str, dict] = {}
    background: list[dict] = []
    with kb_path.open("rb") as source:
        for doc in ijson.items(source, "documents.item"):
            for page in doc.get("pages") or []:
                for block in page.get("blocks") or []:
                    block_id = block.get("block_id")
                    content = (block.get("text") or "").strip()
                    if not block_id or not content:
                        continue
                    item = {
                        "block_id": block_id,
                        "content": content,
                        "source_doc": doc.get("doc_id") or doc.get("doc_name"),
                        "page": page.get("page_no"),
                        "section_path": block.get("section_path"),
                        "type": block.get("block_type") or "paragraph",
                    }
                    if block_id in wanted:
                        collected[block_id] = item
                    elif len(background) < background_target:
                        background.append(item)
    return [collected[bid] for bid in sorted(collected)] + background


def main() -> None:
    parser = argparse.ArgumentParser(description="构建阶段 3 证据评测语料")
    parser.add_argument("--kb", default=str(ROOT / "output" / "knowledge_base.json"))
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "output" / "baseline" / "qa_report.json"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "phase3"))
    parser.add_argument("--background-blocks", type=int, default=985)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold, gold_count = extract_gold(Path(args.baseline))
    corpus = build_corpus(
        Path(args.kb),
        gold,
        background_target=args.background_blocks,
    )
    corpus_path = output_dir / "eval_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as target:
        for item in corpus:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    golden_path = output_dir / "golden_retrieval.json"
    golden = {
        "schema_version": 1,
        "source": "baseline_auto_provisional",
        "note": "gold evidence IDs 取自阶段 0 基线报告，作为 dense 基线对照；上线前需人工校验",
        "cases": [
            {
                "question_id": qid,
                "question": entry["question"],
                "gold_block_ids": entry["gold_block_ids"],
            }
            for qid, entry in gold.items()
        ],
    }
    golden_path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "corpus_blocks": len(corpus),
                "gold_count": gold_count,
                "gold_found": sum(
                    1
                    for item in corpus
                    if item["block_id"]
                    in {
                        bid
                        for entry in gold.values()
                        for bid in entry["gold_block_ids"]
                    }
                ),
                "corpus_path": str(corpus_path),
                "golden_path": str(golden_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
