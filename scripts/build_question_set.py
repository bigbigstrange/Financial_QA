# -*- coding: utf-8 -*-
"""把仓库里的题库（test/*.xlsx）转成统一 JSONL，便于评测脚本复用。

用法：
    python scripts/build_question_set.py                 # 默认导出 merged.xlsx
    python scripts/build_question_set.py --source test/复赛测试集.xlsx

说明：题库只有 ``question_id`` 与 ``question`` 两列，**没有人工标准答案与证据出处**，
因此只能算"运行指标"，不能算"准确率"。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def convert(source: Path, out: Path) -> int:
    df = pd.read_excel(source)
    if "question" not in df.columns:
        raise SystemExit(f"{source} 缺少 question 列，实际列：{list(df.columns)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            question = str(row["question"]).strip()
            if not question:
                continue
            record = {
                "id": f"q{int(row['question_id']):04d}" if "question_id" in df.columns else f"q{n:04d}",
                "question": question,
                "label_status": "unlabeled",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "test" / "merged.xlsx"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "questions.jsonl"))
    args = parser.parse_args()

    count = convert(Path(args.source), Path(args.out))
    print(f"{count} questions -> {args.out}")
    print("注意：题库无人工标准答案/出处，只能评估运行指标，不能评估准确率")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
