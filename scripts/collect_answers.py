# -*- coding: utf-8 -*-
"""把人工标注的 Excel 转成评测数据集，并批量收集系统答案。

用法：
    python scripts/collect_answers.py --convert-only
    python scripts/collect_answers.py --limit 2
    python scripts/collect_answers.py \
        --xlsx eval/annotation/kb50_reviewed.xlsx \
        --out eval/results/answers_kb50.json \
        --dataset eval/annotation/kb50_dataset.jsonl

特性：断点续跑（已答过的跳过）、token 过期自动重登、记录引用命中。
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
DEFAULT_XLSX = ROOT / "eval" / "annotation" / "50问_标注模板.xlsx"
DEFAULT_DATASET = ROOT / "eval" / "annotation" / "answers_dataset.jsonl"
DEFAULT_OUT = ROOT / "eval" / "results" / "answers_50.json"
BASE = "http://127.0.0.1:8002"
USER = ("admin", "demo-admin-2026")


def convert(src: Path, dataset: Path) -> int:
    df = pd.read_excel(src)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dataset.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            record = {
                "id": str(row["id"]),
                "question": str(row["question"]).strip(),
                "gold_answer": str(row.get("gold_answer") or "").strip(),
                "key_facts": str(row.get("key_facts") or "").strip(),
                "source_doc": str(row.get("source_doc") or "").strip(),
            }
            if "chunk_id" in df.columns:
                record["gold_chunk_id"] = str(row.get("chunk_id") or "").strip()
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    results_path = Path(args.out)
    dataset_path = Path(args.dataset)

    count = convert(Path(args.xlsx), dataset_path)
    print(f"dataset: {count} questions -> {dataset_path}")
    if args.convert_only:
        return 0

    questions = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    results: dict[str, dict] = {}
    if results_path.exists():
        for row in json.loads(results_path.read_text(encoding="utf-8")):
            results[row["id"]] = row

    token = login()
    for index, item in enumerate(questions, 1):
        qid = item["id"]
        if qid in results and results[qid].get("answer"):
            continue

        started = time.time()
        status, body = post("/api/runs", {"question": item["question"]}, token, timeout=300)
        if status == 401:  # access token 15 分钟过期
            token = login()
            started = time.time()
            status, body = post("/api/runs", {"question": item["question"]}, token, timeout=300)
        elapsed = round(time.time() - started, 2)

        if status == 200:
            data = json.loads(body)
            evidence = data.get("evidence") or []
            gold_chunk = item.get("gold_chunk_id")
            results[qid] = {
                "id": qid,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "key_facts": item["key_facts"],
                "gold_chunk_id": gold_chunk,
                "answer": data.get("answer") or "",
                "verified": bool(data.get("verified")),
                "confidence": data.get("confidence"),
                "latency_sec": data.get("latency_sec") or elapsed,
                "evidence_count": len(evidence),
                "citation_hit": (
                    any((e.get("block_id") or e.get("chunk_id")) == gold_chunk for e in evidence)
                    if gold_chunk else None
                ),
                "run_id": data.get("run_id"),
                "status": status,
            }
        else:
            results[qid] = {
                "id": qid, "question": item["question"],
                "gold_answer": item["gold_answer"], "key_facts": item["key_facts"],
                "gold_chunk_id": item.get("gold_chunk_id"),
                "answer": "", "verified": False, "confidence": 0,
                "latency_sec": elapsed, "evidence_count": 0,
                "citation_hit": None, "status": status, "error": body[:300],
            }

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=1), encoding="utf-8")
        row = results[qid]
        print(f"[{index}/{len(questions)}] {qid} {status} {row['latency_sec']}s "
              f"verified={row['verified']} ev={row['evidence_count']} hit={row.get('citation_hit')} "
              f":: {row['answer'][:50]}", flush=True)
        time.sleep(args.sleep)

    print(f"done: {len(results)} answered -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
