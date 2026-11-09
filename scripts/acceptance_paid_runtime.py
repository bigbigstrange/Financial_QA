"""Explicit, bounded live HTTP acceptance against the local financial API.

This is exploratory evidence, not a gold-answer accuracy benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from app.core.config import get_settings


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "output" / "pilot_budget.sqlite3"


def _post(url: str, body: dict, *, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers=headers, method="POST")
    try:
        with urlopen(request, timeout=240) as response:
            return json.load(response)
    except HTTPError as exc:
        # The body can contain answer text, but never print request credentials.
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc


def _budget() -> dict:
    with sqlite3.connect(f"file:{LEDGER.as_posix()}?mode=ro", uri=True) as db:
        spent = db.execute("SELECT COALESCE(SUM(total_tokens),0) FROM calls "
                           "WHERE status='settled'").fetchone()[0]
        stopped = db.execute("SELECT stopped_reason FROM state WHERE id=1").fetchone()[0]
    return {"spent_tokens": spent, "stopped_reason": stopped}


def _failed_run(idempotency_key: str) -> dict:
    """Preserve the durable run id when the API correctly returns HTTP 503."""
    path = ROOT / "data_app.db"
    if not path.is_file():
        return {}
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT id,status,error_code FROM runs WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
    return {"run_id": row[0], "status": row[1], "error_code": row[2]} if row else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true", help="required for live model calls")
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--web", action="store_true", help="allow real Tavily routing")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "output" / "acceptance" /
                                "2026-09-29-paid-runtime.json")
    args = parser.parse_args()
    if not args.send:
        parser.error("live calls require --send")
    if len(args.question) > 3:
        parser.error("at most three questions per invocation")
    load_dotenv(ROOT / ".env", override=False)
    if os.environ.get("FINANCE_DISABLE_PAID_API") == "1":
        parser.error("paid model is disabled in this process")
    settings = get_settings()
    base = "http://127.0.0.1:8001/api"
    login = _post(base + "/auth/login", {
        "username": settings.bootstrap_admin_user,
        "password": settings.bootstrap_admin_password,
    })
    token = login["access_token"]
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "exploratory_not_gold",
        "web_enabled": args.web,
        "before_budget": _budget(),
        "cases": [],
    }
    for index, question in enumerate(args.question, 1):
        if _budget()["stopped_reason"]:
            break
        key = (f"paid-acceptance-20260929-{index}-"
               f"{hashlib.sha256(question.encode()).hexdigest()[:12]}")
        try:
            result = _post(base + "/runs", {
                "question": question,
                "idempotency_key": key,
                "use_paid_model": True,
                "use_web_search": args.web,
            }, token=token)
            report["cases"].append({
                "question": question,
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "answer": result.get("answer"),
                "verified": result.get("verified"),
                "route": result.get("route"),
                "trace": result.get("trace"),
                "evidence": result.get("evidence"),
                "latency_sec": result.get("latency_sec"),
            })
        except Exception as exc:
            report["cases"].append({"question": question, "error": str(exc),
                                    **_failed_run(key)})
            break
        finally:
            report["after_budget"] = _budget()
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({
        "report": str(args.report),
        "cases": [{"status": item.get("status"), "run_id": item.get("run_id"),
                   "error": item.get("error"), "verified": item.get("verified"),
                   "route": item.get("route")}
                  for item in report["cases"]],
        "before_budget": report["before_budget"],
        "after_budget": report.get("after_budget", _budget()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
