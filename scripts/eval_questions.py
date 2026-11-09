# -*- coding: utf-8 -*-
"""题库级评测：只测**运行指标**（完成 / 拒答 / 延迟），不测准确率。

为什么不是准确率：`test/*.xlsx` 只有 question_id 与 question 两列，
没有人工标准答案与证据出处（见 docs/baseline 交付审计）。任何"准确率"都必须先有人工标注。

用法：
    python scripts/eval_questions.py --limit 50                 # 演示引擎
    python scripts/eval_questions.py --engine real --limit 50   # 需本地知识库就绪
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_questions(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def demo_runner():
    from app.demo_engine import answer_demo

    def run(question: str, qid: str) -> dict:
        started = time.time()
        result = answer_demo(question, qid)
        return {
            "answered": bool(result["verified"]),
            "latency_ms": round((time.time() - started) * 1000, 2),
        }

    return run


def real_runner():
    from app.services.qa_runtime import QARuntime

    runtime = QARuntime()
    runtime.ensure_loaded()

    def run(question: str, qid: str) -> dict:
        started = time.time()
        result = runtime.answer(question, qid)
        return {
            "answered": bool(result.get("verified")),
            "latency_ms": round((time.time() - started) * 1000, 2),
        }

    return run


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--engine", choices=["demo", "real"], default="demo")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "questions.json"))
    args = parser.parse_args()

    questions = load_questions(Path(args.questions), args.limit)
    runner = demo_runner() if args.engine == "demo" else real_runner()

    rows, latencies = [], []
    for item in questions:
        try:
            outcome = runner(item["question"], item["id"])
            rows.append({**item, **outcome})
            latencies.append(outcome["latency_ms"])
        except Exception as exc:  # 单题失败不应中断整轮评测
            rows.append({**item, "answered": False, "error": str(exc)[:200]})

    answered = sum(1 for r in rows if r.get("answered"))
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": args.engine,
        "total": len(rows),
        "answered": answered,
        "refused": len(rows) - answered,
        "refusal_rate": round(1 - answered / len(rows), 4) if rows else 0.0,
        "errors": sum(1 for r in rows if r.get("error")),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "note": "题库无人工标准答案与证据出处，本报告只含运行指标，不含准确率",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
