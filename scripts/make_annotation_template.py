# -*- coding: utf-8 -*-
"""从题库里挑 50 道"知识库大概率能答"的题，生成人工标注模板。

选择策略：按知识库关键词命中数排序（手机银行 / 网银 / 基金 / 账户 / 证书 …），
同分则按题目长度取中位数附近的，避免全是超长多跳题。

产出：
    eval/annotation/50问_标注模板.xlsx   # 填 gold_answer 用
    eval/annotation/50问.jsonl           # 机器可读
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "eval" / "annotation"

# 与 output/kb_full 加载时使用的关键词保持一致
KEYWORDS = [
    "掌上银行", "手机银行", "网银", "网银盾", "数字证书", "证书在线", "USBKey",
    "客服", "账户", "明细查询", "总资产", "激活", "基金", "申购", "赎回",
    "浦发", "浦东发展", "建行", "邮储", "邮政储蓄", "交通银行", "交行",
    "95580", "40088", "系统升级",
]

TARGET = 50


def score(question: str) -> tuple[int, int]:
    hits = sum(1 for kw in KEYWORDS if kw in question)
    # 次关键字：适中长度优先（20~60 字）
    length = len(question)
    balance = -abs(length - 40)
    return hits, balance


def main() -> int:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    ranked = sorted(rows, key=lambda r: score(r["question"]), reverse=True)

    picked, seen = [], set()
    for item in ranked:
        q = item["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        picked.append(item)
        if len(picked) >= TARGET:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {
            "id": item["id"],
            "question": item["question"],
            "gold_answer": "",
            "key_facts": "",
            "source_doc": "",
            "difficulty": "",
            "notes": "",
        }
        for item in picked
    ])
    xlsx = OUT_DIR / "50问_标注模板.xlsx"
    df.to_excel(xlsx, index=False)

    jsonl = OUT_DIR / "50问.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for item in picked:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"selected {len(picked)} questions")
    print(f"xlsx: {xlsx}")
    print(f"jsonl: {jsonl}")
    print("填写说明：gold_answer 必填；key_facts 建议填（数字/期限/费率等）；source_doc / difficulty 可留空")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
