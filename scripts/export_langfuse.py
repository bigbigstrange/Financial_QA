# -*- coding: utf-8 -*-
"""把一次 run 的 trace 导出到 Langfuse。

优先用官方 SDK（OTLP，Langkfuse v4 接口）；SDK 不可用时退回 v3 HTTP ingestion
（注意：v3 ingestion 在 2026-11-16 之后只接受 score 事件）。

    set LANGFUSE_HOST=https://cloud.langfuse.com
    set LANGFUSE_PUBLIC_KEY=pk-...
    set LANGFUSE_SECRET_KEY=sk-...
    python scripts/export_langfuse.py --trace trace.json

    # 不发送，只预览
    python scripts/export_langfuse.py --trace trace.json --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _sanitize_no_proxy() -> None:
    """httpx 解析 NO_PROXY 中的 [::1] 会抛 InvalidURL，这里剔除该条目。"""
    for var in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(var)
        if not raw:
            continue
        parts = [p for p in raw.split(",") if "::1" not in p]
        os.environ[var] = ",".join(parts)


def load_events(payload: dict) -> list[dict]:
    for key in ("trace", "events", "steps"):
        if isinstance(payload.get(key), list):
            return [e for e in payload[key] if isinstance(e, dict)]
    return []


def export_sdk(payload: dict, *, host: str, public_key: str, secret_key: str, name: str) -> str:
    from langfuse import Langfuse
    from langfuse.types import TraceContext

    events = load_events(payload)
    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    trace_id = client.create_trace_id()

    with client.start_as_current_observation(
        name=name,
        as_type="agent",
        trace_context=TraceContext(trace_id=trace_id),
        input={"question": payload.get("question")},
        metadata={
            "run_id": payload.get("run_id"),
            "agent_mode": payload.get("agent_mode"),
            "nodes": len(events),
        },
    ) as root:
        for event in events:
            with client.start_as_current_observation(
                name=str(event.get("node") or "node"),
                as_type="span",
                input={"decision": event.get("decision")},
                output=event.get("detail") or {},
                metadata={"elapsed_ms": float(event.get("elapsed_ms") or 0)},
            ):
                pass
        root.update(output={"verified": payload.get("verified"), "trace_id": trace_id})

    client.flush()
    return trace_id


def build_v3_batch(payload: dict, trace_name: str) -> dict:
    events = load_events(payload)
    run_id = payload.get("run_id") or "unknown-run"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    batch = [{
        "type": "trace-create",
        "body": {
            "id": run_id, "name": trace_name, "timestamp": base.isoformat(),
            "input": {"question": payload.get("question")},
            "output": {"answer": payload.get("answer"), "verified": payload.get("verified")},
        },
    }]
    cursor = base
    for index, event in enumerate(events):
        ms = float(event.get("elapsed_ms") or 0)
        start, end = cursor, cursor + timedelta(milliseconds=ms)
        cursor = end
        batch.append({
            "type": "span-create",
            "body": {
                "id": f"{run_id}-{index:02d}", "traceId": run_id,
                "name": str(event.get("node") or f"node-{index}"),
                "startTime": start.isoformat(), "endTime": end.isoformat(),
                "input": {"decision": event.get("decision")},
                "output": event.get("detail") or {},
                "metadata": {"elapsed_ms": ms},
            },
        })
    return {"batch": batch}


def export_v3(payload: dict, *, host: str, public_key: str, secret_key: str, name: str) -> int:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/public/ingestion",
        data=json.dumps(build_v3_batch(payload, name)).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Basic " + token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--name", default="financial-qa-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--legacy-v3", action="store_true", help="强制使用已废弃的 v3 HTTP 接口")
    parser.add_argument("--env-file", help="从 key=value 文件读取 LANGFUSE_* 配置")
    args = parser.parse_args()

    if args.env_file:
        for line in Path(args.env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))

    _sanitize_no_proxy()

    payload = json.loads(Path(args.trace).read_text(encoding="utf-8"))

    if args.dry_run:
        print(json.dumps(build_v3_batch(payload, args.name), ensure_ascii=False)[:800])
        print(f"dry-run: {len(load_events(payload))} events")
        return 0

    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not all((host, public_key, secret_key)):
        print("缺少 LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
        return 2

    if args.legacy_v3:
        status = export_v3(payload, host=host, public_key=public_key, secret_key=secret_key, name=args.name)
        print(f"v3 ingestion status={status}（该接口 2026-11-16 起仅接受 score 事件）")
        return 0 if status < 300 else 1

    trace_id = export_sdk(payload, host=host, public_key=public_key, secret_key=secret_key, name=args.name)
    print(f"OTLP via SDK ok, trace_id={trace_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
