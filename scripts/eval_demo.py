# -*- coding: utf-8 -*-
"""离线评测：在示例语料上量化「检索命中」与「拒答正确性」。

用法::

    python scripts/eval_demo.py
    python scripts/eval_demo.py --dataset eval/dataset.jsonl --top-k 3

输出::

    eval/results/metrics.json   # 机器可读
    eval/results/report.md      # 人可读（可直接贴进 README）

设计约束：
- 不调用任何付费 API，可离线复现；
- 指标定义写死在脚本里，避免"换个口径就变好看"；
- 结果文件随仓库提交，任何人可以重跑并对齐。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo_engine import MIN_EVIDENCE_SCORE, answer_demo, load_corpus, search  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def evaluate(dataset: list[dict], top_k: int) -> dict:
    chunks, _ = load_corpus()
    rows = []
    hit = 0
    refusal_ok = 0
    refusal_total = 0
    latencies = []

    for item in dataset:
        started = time.time()
        result = answer_demo(item["question"], item["id"])
        latencies.append(time.time() - started)

        retrieved = [e["chunk_id"] for e in result["evidence"][:top_k]]
        best_score = max((e["score"] for e in result["evidence"]), default=0.0)

        if item["type"] == "factual":
            gold = set(item.get("gold") or [])
            ok = bool(gold & set(retrieved))
            hit += int(ok)
            rows.append({
                "id": item["id"], "type": "factual", "question": item["question"],
                "gold": sorted(gold), "retrieved": retrieved,
                "best_score": best_score, "passed": ok,
            })
        else:
            refusal_total += 1
            ok = result["verified"] is False
            refusal_ok += int(ok)
            rows.append({
                "id": item["id"], "type": "refusal", "question": item["question"],
                "gold": [], "retrieved": retrieved,
                "best_score": best_score, "passed": ok,
            })

    factual_total = len(dataset) - refusal_total
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_chunks": len(chunks),
        "dataset_size": len(dataset),
        "factual_total": factual_total,
        "refusal_total": refusal_total,
        "top_k": top_k,
        "recall_at_k": round(hit / factual_total, 4) if factual_total else 0.0,
        "refusal_accuracy": round(refusal_ok / refusal_total, 4) if refusal_total else 0.0,
        "evidence_threshold": MIN_EVIDENCE_SCORE,
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "passed": hit + refusal_ok,
        "total": len(dataset),
    }
    return {"metrics": metrics, "rows": rows}


def write_report(out_dir: Path, payload: dict) -> None:
    m = payload["metrics"]
    lines = [
        "# 离线评测报告（示例语料）",
        "",
        f"- 生成时间：`{m['generated_at']}`",
        f"- 语料切块：{m['corpus_chunks']}；评测集：{m['dataset_size']} 题"
        f"（事实题 {m['factual_total']} / 拒答题 {m['refusal_total']}）",
        f"- 检索通道：idf 加权词面重叠（无向量库、无大模型）",
        f"- 证据阈值：{m['evidence_threshold']}",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| Recall@{m['top_k']} | **{m['recall_at_k']:.2%}** |",
        f"| 拒答准确率 | **{m['refusal_accuracy']:.2%}** |",
        f"| 平均耗时 | {m['avg_latency_ms']} ms |",
        f"| 通过题数 | {m['passed']} / {m['total']} |",
        "",
        "## 逐题结果",
        "",
        "| ID | 类型 | 问题 | 命中 | 最高分 | 结果 |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['id']} | {row['type']} | {row['question']} | "
            f"{', '.join(row['retrieved']) or '-'} | {row['best_score']:.3f} | "
            f"{'✅' if row['passed'] else '❌'} |"
        )
    lines += [
        "",
        "> 说明：本报告基于 `demo/corpus/` 的**示例语料**，"
        "不代表生产语料上的效果；生产语料评测见 `docs/baseline/` 下的交付审计。",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    payload = evaluate(dataset, args.top_k)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(payload["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(out_dir, payload)

    m = payload["metrics"]
    print(f"dataset={m['dataset_size']} recall@{m['top_k']}={m['recall_at_k']:.2%} "
          f"refusal_acc={m['refusal_accuracy']:.2%} "
          f"latency={m['avg_latency_ms']}ms")
    print(f"written: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
