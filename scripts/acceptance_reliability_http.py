"""Actual local HTTP disconnect/replay, cancel, idempotency and review approval checks."""
import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.config import get_settings


def frames(text):
    result = []
    for frame in text.strip().split("\n\n"):
        row = {}
        for line in frame.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                row[key] = json.loads(value) if key == "data" else value
        if row:
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "output/acceptance/reliability-http-20260930.json")
    args = parser.parse_args()
    s = get_settings()
    base = "http://127.0.0.1:8001/api"
    session = requests.Session()
    login = session.post(base + "/auth/login", json={"username": s.bootstrap_admin_user,
                         "password": s.bootstrap_admin_password}, timeout=10)
    login.raise_for_status()
    session.headers["Authorization"] = "Bearer " + login.json()["access_token"]
    body = {"question": "农业银行2024年一季度营业收入是多少？",
            "idempotency_key": "reliability-" + uuid.uuid4().hex,
            "use_paid_model": False, "use_web_search": False}
    with session.post(base + "/runs/stream", json=body, stream=True, timeout=60) as stream:
        stream.raise_for_status()
        stream.encoding = "utf-8"
        for line in stream.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                run_id = json.loads(line[6:])["run_id"]
                break
    cancelled = session.post(base + f"/runs/{run_id}/cancel", timeout=10)
    cancelled.raise_for_status()
    for _ in range(30):
        result = session.get(base + f"/runs/{run_id}", timeout=10).json()
        if result["status"] not in {"pending", "running"}:
            break
        time.sleep(1)
    assert result["status"] == "cancelled", result["status"]
    assert not result.get("answer")
    replay = session.get(base + f"/runs/{run_id}/events", timeout=10)
    replay.raise_for_status()
    events = frames(replay.text)
    assert sum(e["event"] in {"done", "error", "cancelled"} for e in events) == 1
    last_id = events[0]["id"]
    resumed = session.get(base + f"/runs/{run_id}/events", headers={"Last-Event-ID": last_id}, timeout=10)
    assert frames(resumed.text) == events[1:]
    duplicate = session.post(base + "/runs", json=body, timeout=20)
    duplicate.raise_for_status()
    assert duplicate.json()["run_id"] == run_id
    conflict = session.post(base + "/runs", json={**body, "question": "不同问题"}, timeout=10)
    assert conflict.status_code == 409
    review_run = session.post(base + "/runs", json={
        "question": "你好", "use_paid_model": False, "use_web_search": False,
        "idempotency_key": "review-acceptance-" + uuid.uuid4().hex}, timeout=30)
    review_run.raise_for_status()
    review_id = review_run.json()["run_id"]
    endpoint = base + f"/runs/{review_id}/approvals"
    approved = session.post(endpoint, json={"decision": "accept"}, timeout=10)
    approved.raise_for_status()
    assert approved.json()["ticket_status"] == "open"
    repeated = session.post(endpoint, json={"decision": "accept"}, timeout=10)
    repeated.raise_for_status()
    assert approved.json() == repeated.json()
    assert session.post(endpoint, json={"decision": "decline"}, timeout=10).status_code == 409
    def queued_local(question):
        response = requests.post(base + "/runs", headers=dict(session.headers), json={
            "question": question, "use_paid_model": False, "use_web_search": False,
            "idempotency_key": "queue-acceptance-" + uuid.uuid4().hex}, timeout=90)
        response.raise_for_status()
        result = response.json()
        assert result["status"] == "completed"
        assert result["model_snapshot"]["provider"] == "local"
        return result["run_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        queued_ids = list(pool.map(queued_local, [
            "农业银行2024年一季度营业收入是多少？", "上海银行慧财1231M0502产品期限是多少？"]))
    assert len(set(queued_ids)) == 2
    report = {"run_id": run_id, "disconnect_then_cancel": "passed", "terminal": result["status"],
              "event_replay": "passed", "last_event_id": "passed", "idempotency": "passed",
              "conflicting_key_http_status": conflict.status_code, "paid_model_calls": 0,
              "review_approval": "passed", "review_run_id": review_id,
              "review_ticket_id": approved.json()["ticket_id"], "duplicate_approval": "passed",
              "two_concurrent_local_submissions": "passed", "queued_run_ids": queued_ids}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
