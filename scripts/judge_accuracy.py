# -*- coding: utf-8 -*-
"""用 LLM 当裁判，给 50 题的系统答案打分（0 / 0.5 / 1）。

输入：eval/results/answers_50.json
输出：eval/results/judge_50.json、report.md

裁判只看三样东西：问题、人工标准答案、系统答案。
**不让裁判看到检索证据**，避免"因为引用了就判对"的偏差。

    set ARK_API_KEY=...
    python scripts/judge_accuracy.py
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
ANSWERS = ROOT / "eval" / "results" / "answers_50.json"
OUT_JSON = ROOT / "eval" / "results" / "judge_50.json"
OUT_MD = ROOT / "eval" / "results" / "judge_50.md"

ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODEL = os.environ.get("JUDGE_MODEL", "doubao-seed-evolving")

PROMPT = """你是严格的答案评分员。给定【问题】【标准答案】【关键事实】【系统答案】，
判断系统答案是否正确。

评分标准：
- 1（正确）：核心事实与标准答案一致；数字、期限、条件无冲突。允许表述不同、信息更少但关键点齐全。
- 0.5（部分正确）：方向正确但缺少关键事实，或包含一处不影响结论的小错误。
- 0（错误）：核心事实错误、答非所问、或系统明确拒答（答案里说"无依据/无法回答"）。

只输出一行 JSON，不要解释：
{{"score": 0|0.5|1, "reason": "不超过30字"}}

【问题】{question}
【标准答案】{gold}
【关键事实】{facts}
【系统答案】{answer}
"""


def ark(prompt: str, api_key: str, retries: int = 3) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
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
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError):
            if attempt == retries - 1:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def parse(text: str) -> tuple[float, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return 0.0, "裁判输出无法解析"
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score", 0))
        return max(0.0, min(score, 1.0)), str(data.get("reason", ""))[:60]
    except Exception:
        return 0.0, "裁判输出无法解析"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = (os.environ.get("ARK_API_KEY") or "").strip()
    if not api_key:
        key_file = Path(os.environ.get("ARK_KEY_FILE", r"C:\Users\Ultra7-265K\Desktop\密钥.txt"))
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("缺少 ARK_API_KEY")

    rows = json.loads(ANSWERS.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    judged = []
    for index, row in enumerate(rows, 1):
        prompt = PROMPT.format(
            question=row["question"],
            gold=row["gold_answer"],
            facts=row.get("key_facts") or "（未提供）",
            answer=(row.get("answer") or "（空）")[:1500],
        )
        raw = ark(prompt, api_key)
        score, reason = parse(raw)
        judged.append({**row, "score": score, "reason": reason})
        print(f"[{index}/{len(rows)}] {row['id']} score={score} :: {reason}", flush=True)
        time.sleep(0.5)

    total = len(judged)
    correct = sum(1 for r in judged if r["score"] >= 1.0)
    partial = sum(1 for r in judged if 0 < r["score"] < 1.0)
    wrong = total - correct - partial
    answered = sum(1 for r in judged if r.get("verified"))
    latencies = sorted(r.get("latency_sec") or 0 for r in judged)

    def pct(value: float) -> float:
        return round(value / total * 100, 1) if total else 0.0

    metrics = {
        "total": total,
        "model": MODEL,
        "accuracy": pct(correct),
        "partial_rate": pct(partial),
        "wrong_rate": pct(wrong),
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "answered_rate": pct(answered),
        "avg_latency_sec": round(sum(latencies) / total, 1) if total else 0,
        "p95_latency_sec": round(latencies[int(0.95 * (total - 1))], 1) if total else 0,
    }

    OUT_JSON.write_text(json.dumps({"metrics": metrics, "rows": judged}, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# 50 题真实模型评测（LLM 裁判）",
        "",
        f"- 裁判模型：`{MODEL}`（temperature=0，只看问题/标准答案/系统答案）",
        f"- 题量：{total}｜正确 {correct}｜部分正确 {partial}｜错误 {wrong}",
        f"- **准确率（score=1）**：{metrics['accuracy']}%",
        f"- 部分正确率：{metrics['partial_rate']}%｜错误率：{metrics['wrong_rate']}%",
        f"- 系统标记已回答：{metrics['answered_rate']}%",
        f"- 平均延迟：{metrics['avg_latency_sec']}s｜P95：{metrics['p95_latency_sec']}s",
        "",
        "| ID | 得分 | 判定理由 | 问题 |",
        "|---|---|---|---|",
    ]
    for row in judged:
        lines.append(f"| {row['id']} | {row['score']} | {row['reason']} | {row['question'][:40]} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"written: {OUT_JSON.name}, {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
