"""Bounded, resumable real-model comparison; migrated references are NOT gold.

First run the production HTTP Agent for all 50 original questions. Then generate
a one-shot RAG control using the first retrieval's same evidence, without grading
or verification. Never retry an unknown model outcome or overwrite saved cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from scripts.acceptance_paid_runtime import _post, _budget, _failed_run
from app.core.config import get_settings
from src.agent.prompts import GENERATE_SYSTEM, build_generate_prompt
from src.system.llm_api import DeepSeekAPIClient


def bind_experiment(report, source_hash, code_hash):
    """Legacy/mismatched reports are evidence, not resumable new experiments."""
    identity = {"source_sha256": source_hash, "code_sha256": code_hash}
    if report and report.get("experiment_identity") != identity:
        raise ValueError("legacy or changed experiment; preserve it and choose a new report")
    report["experiment_identity"] = identity
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "output/quality/financial50-live-v1.json")
    args = parser.parse_args()
    if not args.send:
        parser.error("real model calls require --send")
    load_dotenv(ROOT / ".env")
    if os.environ.get("FINANCE_DISABLE_PAID_API") == "1":
        parser.error("paid calls are disabled")
    source = ROOT / "output/quality/financial50-snapshot-candidates.jsonl"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 50
    code_digest = hashlib.sha256()
    for file in sorted([ROOT / "scripts/eval_financial50_live.py"] +
                       [p for folder in ("app", "src") for p in (ROOT / folder).rglob("*.py")]):
        code_digest.update(file.relative_to(ROOT).as_posix().encode() + b"\0" + file.read_bytes())
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    experiment_key = bind_experiment(report, source_hash, code_digest.hexdigest())
    if "cases" not in report:
        report.update({"source_sha256": source_hash, "status": "exploratory_not_gold",
               "comparison": "production_agent_vs_same_first_retrieval_one_shot",
               "snapshot_id": cases[0]["source_snapshot"], "before_budget": _budget(), "cases": []})
    if report["source_sha256"] != source_hash:
        raise ValueError("case version changed; choose a new report")

    def save():
        report["after_budget"] = _budget()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def login():
        settings = get_settings()
        return _post("http://127.0.0.1:8001/api/auth/login", {
            "username": settings.bootstrap_admin_user,
            "password": settings.bootstrap_admin_password})["access_token"]

    token, token_at = login(), time.monotonic()
    indexed = {row["case_id"]: row for row in report["cases"]}
    for case in cases:
        if _budget()["stopped_reason"]:
            break
        row = indexed.get(case["case_id"])
        if row and "agent" in row:
            if row["agent"].get("status") != "completed":
                break
            continue
        row = {"case_id": case["case_id"], "question": case["question"],
               "reference_status": case["review_status"]}
        report["cases"].append(row)
        indexed[case["case_id"]] = row
        if time.monotonic() - token_at > 600:
            token, token_at = login(), time.monotonic()
        key = "financial50-" + experiment_key + "-" + case["case_id"]
        before = _budget()["spent_tokens"]
        try:
            row["agent"] = _post("http://127.0.0.1:8001/api/runs", {
                "question": case["question"], "idempotency_key": key,
                "use_paid_model": True, "use_web_search": case["expected_source"] == "web"}, token=token)
        except Exception as exc:
            row["agent"] = {"error_type": type(exc).__name__, **_failed_run(key)}
        row["agent_tokens"] = _budget()["spent_tokens"] - before
        save()
        print(json.dumps({"case_id": case["case_id"], "arm": "agent",
                          "status": row["agent"].get("status"),
                          "tokens": row["agent_tokens"], "budget": _budget()}, ensure_ascii=False), flush=True)
        if row["agent"].get("status") != "completed":
            break
    if len(report["cases"]) != 50 or any(r["agent"].get("status") != "completed" for r in report["cases"]):
        save()
        return
    client = DeepSeekAPIClient()
    checkpoint_path = Path(os.getenv("FINANCE_GRAPH_CHECKPOINT_PATH") or ROOT / "output/run_checkpoints.sqlite3")
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        for row in report["cases"]:
            if "baseline" in row or _budget()["stopped_reason"]:
                if row.get("baseline", {}).get("status") == "uncertain" or _budget()["stopped_reason"]:
                    break
                continue
            history = list(saver.list({"configurable": {"thread_id": row["agent"]["run_id"]}}))
            evidence = []
            for item in reversed(history):
                state = item.checkpoint.get("channel_values", {})
                if state.get("documents"):
                    evidence = state["documents"][:8]
                    break
            before, started = _budget()["spent_tokens"], time.monotonic()
            # Persist intent before sending; a killed evaluator cannot silently
            # repeat a model call whose output/accounting was not saved.
            row["baseline"] = {"status": "uncertain", "evidence": evidence}
            save()
            try:
                answer = client.chat([
                    {"role": "system", "content": GENERATE_SYSTEM},
                    {"role": "user", "content": build_generate_prompt(row["question"], evidence)}],
                    max_tokens=1200, node="eval_baseline")
                row["baseline"].update(status="completed", answer=answer)
            except Exception as exc:
                row["baseline"]["error_type"] = type(exc).__name__
            row["baseline"].update(tokens=_budget()["spent_tokens"] - before,
                                    latency_sec=round(time.monotonic() - started, 3))
            save()
            print(json.dumps({"case_id": row["case_id"], "arm": "baseline",
                              "status": row["baseline"]["status"], "budget": _budget()}, ensure_ascii=False), flush=True)
            if row["baseline"]["status"] != "completed":
                break
    report["execution_complete"] = all(r.get("baseline", {}).get("status") == "completed" for r in report["cases"])
    save()


if __name__ == "__main__":
    main()
