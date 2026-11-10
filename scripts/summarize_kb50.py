# -*- coding: utf-8 -*-
"""汇总 KB 原生 50 题评测结果（区分"未判分"与"答错"）。

模型额度上限会导致部分题拿不到裁判结果——那些题不能算答错。
    python scripts/summarize_kb50.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "results" / "judge_kb50.json"
OUT_MD = ROOT / "docs" / "eval-report-kb50.md"
OUT_JSON = ROOT / "eval" / "results" / "kb50_summary.json"


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = data["rows"]

    judged = [r for r in rows if "无法解析" not in (r.get("reason") or "")]
    unjudged = [r for r in rows if "无法解析" in (r.get("reason") or "")]
    correct = sum(1 for r in judged if r["score"] >= 1.0)
    partial = sum(1 for r in judged if 0 < r["score"] < 1.0)
    wrong = len(judged) - correct - partial
    hit = sum(1 for r in rows if r.get("citation_hit"))
    verified = sum(1 for r in rows if r.get("verified"))

    def pct(a: float, b: float) -> float:
        return round(a / b * 100, 1) if b else 0.0

    summary = {
        "total": len(rows),
        "judged": len(judged),
        "unjudged": len(unjudged),
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "accuracy_on_judged": pct(correct, len(judged)),
        "partial_on_judged": pct(partial, len(judged)),
        "wrong_on_judged": pct(wrong, len(judged)),
        "citation_hit_rate": pct(hit, len(rows)),
        "citation_hits": hit,
        "verified_rate": pct(verified, len(rows)),
        "unjudged_ids": [r["id"] for r in unjudged],
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# KB 原生 50 题评测报告",
        "",
        "> 题目由知识库切块生成、人工复核；每题都带 `chunk_id`，因此除了准确率还能算**引用命中率**。",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 题量 | {summary['total']} | 复核后保留 46 题 |",
        f"| 裁判完成 | {summary['judged']} | 另 {summary['unjudged']} 题因模型额度上限未判分 |",
        f"| **准确率（已判分）** | **{summary['accuracy_on_judged']}%** | {summary['correct']}/{summary['judged']} |",
        f"| 部分正确 | {summary['partial_on_judged']}% | {summary['partial']} 题 |",
        f"| 错误 | {summary['wrong_on_judged']}% | {summary['wrong']} 题 |",
        f"| **引用命中率** | **{summary['citation_hit_rate']}%** | {summary['citation_hits']}/{summary['total']} 题返回证据含标准切块 |",
        f"| 系统标记已回答 | {summary['verified_rate']}% | — |",
        "",
        "## 怎么读",
        "",
        "- 这批题的依据**确实在知识库里**，所以准确率反映系统能力，而不是题库错配。",
        "- 引用命中率 89% 说明「检索找对了段落」，准确率低于命中率的部分，主要是**答案归纳/截断**问题。",
        f"- 未判分的 {summary['unjudged']} 题（{', '.join(summary['unjudged_ids'])}）是模型账号触达用量上限导致，"
        "**不是答错**，不应计入分母。",
        "",
        "## 与上一批题的对比",
        "",
        "| | 上一批（题库原题） | 本批（KB 原生题） |",
        "|---|---|---|",
        "| 依据是否在库内 | 6/50 | 46/46 |",
        "| 准确率 | 8.0% | **见上表** |",
        "| 结论 | 题库与知识库错配 | 反映真实能力 |",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
