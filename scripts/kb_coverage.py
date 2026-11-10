# -*- coding: utf-8 -*-
"""先算"知识库覆盖"，再谈准确率。

背景：题库与人工资料并非同源——有些题的知识库根本没有对应文档。
这类题系统**应该拒答**，把它算进"答错"会低估系统。

做法：把每道题的 key_facts 拆成条目，检查条目里的关键字符串是否出现在任何切块中。
输出：eval/results/kb_coverage.json

    python scripts/kb_coverage.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform\output\kb_full\chunks.jsonl")
DATASET = ROOT / "eval" / "annotation" / "answers_dataset.jsonl"
OUT = ROOT / "eval" / "results" / "kb_coverage.json"

SPLIT = re.compile(r"[；;，,、]|(?<=\d)\s+")


def load_text() -> str:
    parts = []
    with KB.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts.append(str(row.get("text") or row.get("content") or ""))
    return "\n".join(parts)


def facts(text: str) -> list[str]:
    items = [x.strip() for x in SPLIT.split(text or "") if x.strip()]
    # 只保留有区分度的条目：长度 >= 4 且含中文/数字
    return [x for x in items if len(x) >= 4]


def main() -> int:
    corpus = load_text()
    print(f"corpus chars: {len(corpus):,}")

    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        items = facts(item.get("key_facts", ""))
        if not items:
            items = facts(item.get("gold_answer", ""))[:5]
        found = [x for x in items if x in corpus]
        ratio = len(found) / len(items) if items else 0.0
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "facts_total": len(items),
            "facts_found": len(found),
            "coverage": round(ratio, 3),
            "covered": ratio >= 0.3,
            "missing_examples": [x for x in items if x not in corpus][:3],
        })

    covered = [r for r in rows if r["covered"]]
    summary = {
        "total": len(rows),
        "covered": len(covered),
        "uncovered": len(rows) - len(covered),
        "covered_ids": [r["id"] for r in covered],
        "uncovered_ids": [r["id"] for r in rows if not r["covered"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
