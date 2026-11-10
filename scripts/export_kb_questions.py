# -*- coding: utf-8 -*-
"""把 KB 原生题导出成人工复核用的 Excel。

    python scripts/export_kb_questions.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "annotation" / "kb_questions.jsonl"
OUT = ROOT / "eval" / "annotation" / "KB原生50题_标注模板.xlsx"


def main() -> int:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for index, row in enumerate(rows, 1):
        records.append({
            "id": f"k{index:03d}",
            "question": row["question"],
            "gold_answer": row.get("gold_answer", ""),
            "key_facts": row.get("key_facts", ""),
            "source_doc": row.get("source_doc", ""),
            "page": row.get("page", ""),
            "chunk_id": row.get("chunk_id", ""),
            "备注（复核意见）": "",
        })

    df = pd.DataFrame(records)
    df.to_excel(OUT, index=False)
    print(f"{len(records)} questions -> {OUT}")
    print("复核要点：答案是否只依赖出处片段；数字是否与原文一致；问题是否像真实业务提问")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
