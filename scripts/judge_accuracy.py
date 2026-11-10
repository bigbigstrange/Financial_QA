# -*- coding: utf-8 -*-
"""用 LLM 当裁判，给系统答案打分（0 / 0.5 / 1），并统计引用命中率。

    python scripts/judge_accuracy.py
    python scripts/judge_accuracy.py --answers eval/results/answers_kb50.json \
        --out-json eval/results/judge_kb50.json --out-md docs/eval-report-kb50.md
    # 只补判"裁判输出无法解析"的题（断点续判）
    python scripts/judge_accuracy.py --answers ... --out-json ... --only-failed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODEL = os.environ.get("JUDGE_MODEL", "doubao-seed-evolving")

PROMPT = """你是严格的答案评分员。给定【问题】【标准答案】【关键事实】【系统答案】，
判断系统答案是否正确。

评分标准：
- 1（正确）：核心事实与标准答案一致；数字、期限、条件无冲突。允许表述不同、信息更少但关键点齐全。
- 0.5（部分正确）：方向正确但缺少关键事实，或包含一处不影响结论的小错误。
- 0（错误）：核心事实错误、答非所问、或系统明确拒答（答案里说"无依据/无法回答"）。

只输出一行 JSON，不要解释，不要加代码块：
{{"score": 0|0.5|1, "reason": "不超过30字"}}

【问题】{question}
【标准答案】{gold}
【关键事实】{facts}
【系统答案】{answer}
"""

RETRY_HINT = "\n\n注意：上一次输出无法解析。请只输出一个 JSON 对象，score 只能是 0、0.5 或 1。"


def ark(prompt: str, api_key: str, retries: int = 5, max_tokens: int = 400) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            ARK_URL, data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=90)
            data = json.loads(resp.read().decode("utf-8", "ignore"))
            return data["choices"][0]["message"].get("content", "")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError,
                KeyError, json.JSONDecodeError):
            if attempt == retries - 1:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def parse(text: str) -> tuple[float, str] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score"))
    except Exception:
        return None
    return max(0.0, min(score, 1.0)), str(data.get("reason", ""))[:60]


def judge_one(row: dict, api_key: str) -> tuple[float, str]:
    prompt = PROMPT.format(
        question=row["question"],
        gold=row["gold_answer"],
        facts=row.get("key_facts") or "（未提供）",
        answer=(row.get("answer") or "（空）")[:1500],
    )
    for attempt in range(2):
        raw = ark(prompt + (RETRY_HINT if attempt else ""), api_key)
        parsed = parse(raw)
        if parsed:
            return parsed
        time.sleep(1)
    return 0.0, "裁判输出无法解析"


def metrics_of(judged: list[dict]) -> dict:
    total = len(judged)
    correct = sum(1 for r in judged if r["score"] >= 1.0)
    partial = sum(1 for r in judged if 0 < r["score"] < 1.0)
    wrong = total - correct - partial
    answered = sum(1 for r in judged if r.get("verified"))
    hits = [r for r in judged if r.get("citation_hit") is not None]
    hit_count = sum(1 for r in hits if r.get("citation_hit"))
    latencies = sorted(r.get("latency_sec") or 0 for r in judged)

    def pct(value: float, base: float = total) -> float:
        return round(value / base * 100, 1) if base else 0.0

    return {
        "total": total,
        "model": MODEL,
        "accuracy": pct(correct),
        "partial_rate": pct(partial),
        "wrong_rate": pct(wrong),
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "answered_rate": pct(answered),
        "citation_hit_rate": pct(hit_count, len(hits)) if hits else None,
        "citation_base": len(hits),
        "avg_latency_sec": round(sum(latencies) / total, 1) if total else 0,
        "p95_latency_sec": round(latencies[int(0.95 * (total - 1))], 1) if total else 0,
    }


def write_markdown(judged: list[dict], metrics: dict, out_md: Path) -> None:
    lines = [
        "# 真实模型评测（LLM 裁判）",
        "",
        f"- 裁判模型：`{MODEL}`（temperature=0，只看问题/标准答案/系统答案）",
        f"- 题量：{metrics['total']}｜正确 {metrics['correct']}｜部分正确 {metrics['partial']}｜错误 {metrics['wrong']}",
        f"- **准确率（score=1）**：{metrics['accuracy']}%",
        f"- 部分正确率：{metrics['partial_rate']}%｜错误率：{metrics['wrong_rate']}%",
        f"- 系统标记已回答：{metrics['answered_rate']}%",
        (f"- **引用命中率**：{metrics['citation_hit_rate']}%"
         f"（{int(metrics['citation_hit_rate'] * metrics['citation_base'] / 100)}/{metrics['citation_base']} 题的返回证据包含标准切块）"
         if metrics["citation_hit_rate"] is not None else "- 引用命中率：未标注 chunk_id，跳过"),
        f"- 平均延迟：{metrics['avg_latency_sec']}s｜P95：{metrics['p95_latency_sec']}s",
        "",
        "| ID | 得分 | 引用命中 | 判定理由 | 问题 |",
        "|---|---|---|---|---|",
    ]
    for row in judged:
        hit = "是" if row.get("citation_hit") else ("—" if row.get("citation_hit") is None else "否")
        lines.append(f"| {row['id']} | {row['score']} | {hit} | {row['reason']} | {row['question'][:40]} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default=str(ROOT / "eval" / "results" / "answers_50.json"))
    parser.add_argument("--out-json", default=str(ROOT / "eval" / "results" / "judge_50.json"))
    parser.add_argument("--out-md", default=str(ROOT / "eval" / "results" / "judge_50.md"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-failed", action="store_true")
    args = parser.parse_args()

    api_key = (os.environ.get("ARK_API_KEY") or "").strip()
    if not api_key:
        key_file = Path(os.environ.get("ARK_KEY_FILE", r"C:\Users\Ultra7-265K\Desktop\密钥.txt"))
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("缺少 ARK_API_KEY")

    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    if args.limit:
        answers = answers[: args.limit]

    out_json = Path(args.out_json)
    existing: dict[str, dict] = {}
    if args.only_failed and out_json.exists():
        for row in json.loads(out_json.read_text(encoding="utf-8"))["rows"]:
            existing[row["id"]] = row

    todo = answers
    if args.only_failed:
        todo = [row for row in answers
                if "无法解析" in (existing.get(row["id"], {}).get("reason") or "")]

    updated: dict[str, dict] = dict(existing)
    for index, row in enumerate(todo, 1):
        score, reason = judge_one(row, api_key)
        updated[row["id"]] = {**row, "score": score, "reason": reason}
        print(f"[{index}/{len(todo)}] {row['id']} score={score} hit={row.get('citation_hit')} :: {reason}", flush=True)
        time.sleep(2.5)

    judged = [updated.get(row["id"], row) for row in answers]
    metrics = metrics_of(judged)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"metrics": metrics, "rows": judged}, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(judged, metrics, Path(args.out_md))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"written: {out_json.name}, {Path(args.out_md).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
