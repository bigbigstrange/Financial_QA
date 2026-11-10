# -*- coding: utf-8 -*-
"""合并"裁判打分"与"知识库覆盖"，输出分组评测报告。

分组逻辑：
- covered：key_facts 在知识库中能查到的题 → 考察**答得对不对**
- uncovered：知识库里根本没有依据的题 → 考察**该不该拒答**

    python scripts/eval_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "eval" / "results" / "judge_50.json"
COVERAGE = ROOT / "eval" / "results" / "kb_coverage.json"
OUT_MD = ROOT / "docs" / "eval-report.md"
OUT_JSON = ROOT / "eval" / "results" / "eval_summary.json"


def group(rows: list[dict], ids: set[str]) -> dict:
    subset = [r for r in rows if r["id"] in ids]
    total = len(subset)
    if not total:
        return {"total": 0}
    correct = sum(1 for r in subset if r["score"] >= 1.0)
    partial = sum(1 for r in subset if 0 < r["score"] < 1.0)
    refused = sum(1 for r in subset if not r.get("verified"))
    return {
        "total": total,
        "correct": correct,
        "partial": partial,
        "accuracy": round(correct / total * 100, 1),
        "partial_rate": round(partial / total * 100, 1),
        "refusal_rate": round(refused / total * 100, 1),
    }


def main() -> int:
    judge = json.loads(JUDGE.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    rows = judge["rows"]
    covered_ids = set(coverage["summary"]["covered_ids"])

    all_g = group(rows, {r["id"] for r in rows})
    covered = group(rows, covered_ids)
    uncovered = group(rows, {r["id"] for r in rows} - covered_ids)

    summary = {
        "judge_model": judge["metrics"]["model"],
        "overall": all_g,
        "covered": covered,
        "uncovered": uncovered,
        "note": "covered 考察准确率；uncovered 考察是否在无依据时拒答",
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 真实模型评测报告",
        "",
        f"> 裁判模型：`{judge['metrics']['model']}`（temperature=0，只看问题 / 标准答案 / 系统答案）",
        "",
        "## 先说结论",
        "",
        f"这套知识库（30,406 切块）与这 50 道题**并不同源**：只有 **{covered['total']} 题** 的关键事实能在库中查到，"
        f"其余 **{uncovered['total']} 题**库里没有依据。",
        "",
        "所以要把两类题分开看：",
        "",
        "| 分组 | 题数 | 准确率 | 部分正确 | 拒答率 |",
        "|---|---|---|---|---|",
        f"| 全部 | {all_g['total']} | {all_g['accuracy']}% | {all_g['partial_rate']}% | {all_g['refusal_rate']}% |",
        f"| **库内有依据**（考察答得对） | {covered['total']} | **{covered['accuracy']}%** | {covered['partial_rate']}% | {covered['refusal_rate']}% |",
        f"| **库内无依据**（考察该拒答） | {uncovered['total']} | — | — | **{uncovered['refusal_rate']}%** |",
        "",
        "## 怎么读这张表",
        "",
        "- 「库内无依据」这一组的正确行为**就是拒答**；拒答率越高说明系统越不容易编。",
        "- 「库内有依据」这组才适合谈准确率，但样本量小，**不能当作总体准确率**。",
        "- 想要一个有统计意义的准确率，需要**从知识库文档本身出题**，而不是沿用这份题库。",
        "",
        "## 逐题明细",
        "",
        "| ID | 分组 | 得分 | 系统是否拒答 | 判定理由 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        grp = "库内有依据" if row["id"] in covered_ids else "库内无依据"
        lines.append(
            f"| {row['id']} | {grp} | {row['score']} | "
            f"{'是' if not row.get('verified') else '否'} | {row['reason']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
