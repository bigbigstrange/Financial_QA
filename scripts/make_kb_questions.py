# -*- coding: utf-8 -*-
"""从知识库切块生成 50 道"KB 原生题"（题目 + 草稿答案 + 出处）。

与上一批题的区别：这批题的**依据一定在知识库里**，跑出来的准确率才有意义。

    set ARK_API_KEY=...
    python scripts/make_kb_questions.py --limit 50

产出：
    eval/annotation/kb_questions.jsonl   # 含 chunk_id / 出处，可断点续跑
    eval/annotation/KB原生50题_标注模板.xlsx
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform\output\kb_full\chunks.jsonl")
OUT_JSONL = ROOT / "eval" / "annotation" / "kb_questions.jsonl"
OUT_XLSX = ROOT / "eval" / "annotation" / "KB原生50题_标注模板.xlsx"

ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODEL = os.environ.get("GEN_MODEL", "doubao-seed-evolving")

PROMPT = """你是金融业务出题人。请根据下面的【文档片段】出一道具业务感的问题，并给出答案。

要求：
1. 问题必须**只能**从该片段得到答案，不要引入片段外的信息；
2. 问题要像业务人员真实会问的（例如"XX 的限额是多少""XX 需要几步"），不要写"根据上文"；
3. 答案 2-4 句，必须包含片段中的关键数字/条件；
4. key_facts 用分号分隔 2-4 个关键事实（数字、期限、条件）；
5. 只输出一行 JSON：{{"question": "...", "answer": "...", "key_facts": "..."}}

【文档片段】
{content}
"""


def ark(prompt: str, api_key: str, retries: int = 3) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            ARK_URL, data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read().decode("utf-8", "ignore"))
            return data["choices"][0]["message"].get("content", "")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError):
            if attempt == retries - 1:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def candidates(limit_docs: int = 60) -> list[dict]:
    """挑信息密度高、可独立成题的切块，每个文档最多 2 条。"""
    by_doc: dict[str, list[dict]] = {}
    with KB.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get("content") or "").strip()
            if not (180 <= len(text) <= 420):
                continue
            if len(re.findall(r"[。；]", text)) < 2:
                continue
            if not re.search(r"\d", text):
                continue
            if "under_min_chars" in str(row.get("quality_flags") or ""):
                continue
            doc = str(row.get("source_doc") or "")
            if not doc:
                continue
            bucket = by_doc.setdefault(doc, [])
            if len(bucket) < 2:
                bucket.append(row)
            if len(by_doc) >= limit_docs:
                break

    pool = [row for rows in by_doc.values() for row in rows]
    random.Random(20261110).shuffle(pool)
    return pool


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not all(data.get(k) for k in ("question", "answer")):
        return None
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    api_key = (os.environ.get("ARK_API_KEY") or "").strip()
    if not api_key:
        key_file = Path(r"C:\Users\Ultra7-265K\Desktop\密钥.txt")
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("缺少 ARK_API_KEY")

    done: dict[str, dict] = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["chunk_id"]] = row

    pool = candidates()
    picked = 0
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a", encoding="utf-8") as fh:
        for row in pool:
            if picked >= args.limit:
                break
            chunk_id = str(row.get("chunk_id") or row.get("block_id"))
            if chunk_id in done:
                picked += 1
                continue
            content = str(row.get("content") or "")[:900]
            raw = ark(PROMPT.format(content=content), api_key)
            data = parse(raw)
            if not data:
                print(f"skip {chunk_id}: 生成失败", flush=True)
                continue
            record = {
                "chunk_id": chunk_id,
                "question": data["question"].strip(),
                "gold_answer": data["answer"].strip(),
                "key_facts": str(data.get("key_facts") or "").strip(),
                "source_doc": row.get("source_doc"),
                "section_path": row.get("section_path"),
                "page": row.get("page"),
                "evidence": content[:400],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            picked += 1
            print(f"[{picked}/{args.limit}] {record['question'][:44]}", flush=True)
            time.sleep(0.3)

    print(f"done: {picked} questions -> {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
