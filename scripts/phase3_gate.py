#!/usr/bin/env python3
"""Retrieval quality gate for the phase 3 hybrid baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GATE_DEFAULTS = {
    "hybrid+rerank.recall_at_10": 0.5,
    "hybrid+rerank.mrr": 0.4,
    "hybrid+rerank.ndcg_at_10": 0.4,
}


def evaluate_gate(
    report: Dict[str, Any],
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    thresholds = {**GATE_DEFAULTS, **(thresholds or {})}
    strategies = report.get("strategies") or {}
    if "hybrid+rerank" not in strategies:
        return {
            "passed": False,
            "checks": [],
            "error": "report missing hybrid+rerank strategy",
        }
    target = strategies["hybrid+rerank"]
    checks = []
    for key, threshold in thresholds.items():
        strategy_name, metric = key.split(".", 1)
        value = float(target.get(metric) or 0.0)
        checks.append(
            {
                "strategy": strategy_name,
                "metric": metric,
                "threshold": threshold,
                "value": value,
                "passed": value >= threshold,
            }
        )
    dense = strategies.get("dense") or {}
    hybrid = strategies.get("hybrid") or {}
    checks.append(
        {
            "strategy": "ab",
            "metric": "hybrid_vs_dense_recall@10",
            "threshold": "hybrid >= dense",
            "value": float(hybrid.get("recall_at_10") or 0.0),
            "passed": float(hybrid.get("recall_at_10") or 0.0)
            >= float(dense.get("recall_at_10") or 0.0),
        }
    )
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3 检索质量门禁")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = evaluate_gate(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
