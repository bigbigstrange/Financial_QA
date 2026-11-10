# -*- coding: utf-8 -*-
"""把人工标注的 Excel 转成评测数据集，并批量收集系统答案。

用法：
    # 1) 只转换
    python scripts/collect_answers.py --convert-only

    # 2) 跑前 2 题（冒烟）
    python scripts/collect_answers.py --limit 2

    # 3) 全量 50 题（可断点续跑：已答过的会跳过）
    python scripts/collect_answers.py

产物：
    eval/annotation/answers_dataset.jsonl    # 题库 + 金标准
    eval/results/answers_50.json             # 系统输出（含耗时、证据数）
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_XLSX = ROOT / "eval" / "annotation" / "50问_标注模板.xlsx"
DATASET = ROOT / "eval" / "annotation" / "answers_dataset.jsonl"
RESULTS = ROOT / "eval" / "results" / "answers_50.json"
BASE = "http://127.0.0.1:8002"
USER = ("admin", "demo-admin-2026")


def convert() -> int:
    df = pd.read_excel(SRC_XLSX)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with DATASET.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            record = {
                "id": str(row["id"]),
                "question": str(row["question"]).strip(),
                "gold_answer": str(row.get("gold_answer") or "").strip(),
                "key_facts": str(row.get("key_facts") or "").strip(),
                "source_doc": str(row.get("source_doc") or "").strip(),
            }
            if not record["gold_answer"]:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def post(path: str, payload: dict | None = None, token: str | None = None, timeout: int = 300):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        BASE + path, data=data, headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "ignore")


def login() -> str:
    status, body = post("/api/auth/login", {"username": USER[0], "password": USER[1]})
    if status != 200:
        raise SystemExit(f"登录失败 {status}: {body[:200]}")
    return json.loads(body)["access_token"]


def load_done() -> dict:
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        return {row["id"]: row for row in data}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    count = convert()
    print(f"dataset: {count} questions -> {DATASET}")
    if args.convert_only:
        return 0

    questions = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    done = load_done()
    token = login()
    results = dict(done)

    for index, item in enumerate(questions, 1):
        qid = item["id"]
        if qid in results and results[qid].get("answer"):
            continue
        started = time.time()
        status, body = post("/api/runs", {"question": item["question"]}, token, timeout=300)
        if status == 401:  # access token 只有 15 分钟，过期就重新登录重试一次
            token = login()
            started = time.time()
            status, body = post("/api/runs", {"question": item["question"]}, token, timeout=300)
        elapsed = round(time.time() - started, 2)
        if status == 200:
            data = json.loads(body)
            results[qid] = {
                "id": qid,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "key_facts": item["key_facts"],
                "answer": data.get("answer") or "",
                "verified": bool(data.get("verified")),
                "confidence": data.get("confidence"),
                "latency_sec": data.get("latency_sec") or elapsed,
                "evidence_count": len(data.get("evidence") or []),
                "run_id": data.get("run_id"),
                "status": status,
            }
        else:
            results[qid] = {
                "id": qid, "question": item["question"],
                "gold_answer": item["gold_answer"], "key_facts": item["key_facts"],
                "answer": "", "verified": False, "confidence": 0,
                "latency_sec": elapsed, "evidence_count": 0,
                "status": status, "error": body[:300],
            }
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=1), encoding="utf-8")
        row = results[qid]
        print(f"[{index}/{len(questions)}] {qid} {status} {row['latency_sec']}s "
              f"verified={row['verified']} ev={row['evidence_count']} :: {row['answer'][:60]}", flush=True)
        time.sleep(args.sleep)

    print(f"done: {len(results)} answered -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
