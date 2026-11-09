"""Exercise actual HTTP runs/trace with local configured bootstrap credentials."""
import json
import argparse
from pathlib import Path
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.config import get_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--report", type=Path, default=ROOT / "output/kb_v3/platform_channel_smoke.json")
    parser.add_argument("--local-only", action="store_true",
                        help="Disable paid model and Web search for this smoke run")
    args = parser.parse_args()
    base = args.base_url
    settings = get_settings()
    session = requests.Session()
    login = session.post(base + "/api/auth/login", json={"username": settings.bootstrap_admin_user,
                            "password": settings.bootstrap_admin_password}, timeout=30)
    login.raise_for_status()
    session.headers["Authorization"] = "Bearer " + login.json()["access_token"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    switches = {"use_paid_model": False, "use_web_search": False} if args.local_only else {}
    report = {"base_url": base, "local_only": args.local_only, "cases": []}
    for question in ["你好", "农业银行2024年一季度营业收入和归母净利润分别是多少？", "火星殖民贷款利率是多少？"]:
        start = time.monotonic()
        response = session.post(base + "/api/runs", json={"question": question, **switches}, timeout=300)
        result = response.json()
        row = {"question": question, "http_status": response.status_code, "result": result,
               "elapsed_seconds": round(time.monotonic() - start, 2)}
        if result.get("run_id"):
            trace = session.get(base + f"/api/runs/{result['run_id']}/trace", timeout=30)
            row["trace_http_status"] = trace.status_code
            row["trace"] = trace.json()
            row["evidence_checks"] = []
            for item in result.get("evidence") or []:
                detail = session.get(base + "/api/evidence/" + item["evidence_id"], timeout=30)
                row["evidence_checks"].append({"id": item["evidence_id"], "http_status": detail.status_code,
                    "text_matches": detail.status_code == 200 and detail.json().get("text") == item.get("text")})
        report["cases"].append(row)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"question": question, "http_status": response.status_code,
            "route": result.get("route"), "evidence": len(result.get("evidence") or []),
            "answer_preview": (result.get("answer") or "")[:180], "elapsed": row["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    chat = session.post(base + "/api/chat", json={"question": "你好", **switches}, timeout=300)
    chat_result = chat.json()
    report["chat"] = {"http_status": chat.status_code, "run_id": chat_result.get("run_id"),
                      "conversation_id": chat_result.get("conversation_id")}
    if chat.status_code == 200:
        history = session.get(base + f"/api/conversations/{chat_result['conversation_id']}", timeout=30)
        report["chat"]["history_http_status"] = history.status_code
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if chat.status_code != 200 or report["chat"].get("history_http_status") != 200:
        raise SystemExit(2)
    if any(c["http_status"] != 200 or c.get("trace_http_status") != 200
           or any(not e["text_matches"] for e in c.get("evidence_checks", [])) for c in report["cases"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
