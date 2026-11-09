# -*- coding: utf-8 -*-
"""把一次 run 的 trace 导出到 Langfuse（可选集成）。

默认 dry-run，只打印将发送的 payload，便于在没有 Langfuse 实例时验证格式。

    # 预览
    python scripts/export_langfuse.py --trace trace.json --dry-run

    # 实际发送
    set LANGFUSE_HOST=https://cloud.langfuse.com
    set LANGFUSE_PUBLIC_KEY=pk-...
    set LANGFUSE_SECRET_KEY=sk-...
    python scripts/export_langfuse.py --trace trace.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


def build_batch(payload: dict, *, trace_name: str = "financial-qa-run") -> dict:
    trace = payload.get("trace") or payload.get("events") or []
    run_id = payload.get("run_id") or "unknown-run"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    batch: list[dict] = [{
        "type": "trace-create",
        "body": {
            "id": run_id,
            "name": trace_name,
            "timestamp": base.isoformat(),
            "input": {"question": payload.get("question")},
            "output": {"answer": payload.get("answer"), "verified": payload.get("verified")},
            "metadata": {
                "agent_mode": payload.get("agent_mode"),
                "route": payload.get("route"),
                "retry_retrieve": payload.get("retry_retrieve"),
                "retry_generate": payload.get("retry_generate"),
                "nodes": len(trace),
            },
        },
    }]

    cursor = base
    for index, event in enumerate(trace):
        ms = float(event.get("elapsed_ms") or 0)
        start = cursor
        end = cursor + timedelta(milliseconds=ms)
        cursor = end
        batch.append({
            "type": "span-create",
            "body": {
                "id": f"{run_id}-{index:02d}",
                "traceId": run_id,
                "name": str(event.get("node") or f"node-{index}"),
                "startTime": start.isoformat(),
                "endTime": end.isoformat(),
                "input": {"decision": event.get("decision")},
                "output": event.get("detail") or {},
                "metadata": {"elapsed_ms": ms},
            },
        })
    return {"batch": batch}


def post(host: str, public_key: str, secret_key: str, body: dict) -> tuple[int, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/public/ingestion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Basic " + token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode("utf-8", "ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--name", default="financial-qa-run")
    args = parser.parse_args()

    payload = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    body = build_batch(payload, trace_name=args.name)

    if args.dry_run:
        print(json.dumps(body, ensure_ascii=False, indent=2)[:1200])
        print(f"... 共 {len(body['batch'])} 条（1 trace + {len(body['batch']) - 1} spans）")
        return 0

    host = os.environ.get("LANGFUSE_HOST")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not all((host, public_key, secret_key)):
        print("缺少 LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，请先设置环境变量")
        return 2

    status, text = post(host, public_key, secret_key, body)
    print(f"status={status} body={text[:200]}")
    return 0 if status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
