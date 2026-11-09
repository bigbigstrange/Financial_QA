"""Read saved runs and graph results without model calls; never report accuracy as verified."""
import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "output/quality/financial50-live-v1.json"
report = json.loads(source.read_text(encoding="utf-8"))
rows = report["cases"]
completed = [row for row in rows if row["agent"].get("status") == "completed"]
latencies = sorted(row["agent"]["latency_sec"] for row in completed)
summary = {
    "status": "budget_limited_exploratory_not_gold",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "candidate_sha256": report["source_sha256"], "snapshot_id": report["snapshot_id"],
    "scheduled_questions": 50, "completed": len(completed),
    "failed": len(rows) - len(completed), "not_started": 50 - len(rows),
    "failure_codes": dict(Counter(row["agent"].get("error_code") for row in rows if row not in completed)),
    "automated_verified_flags": sum(row["agent"].get("verified") is True for row in completed),
    "answer_accuracy": None, "human_reviewed_gold": False,
    "baseline_completed": sum(row.get("baseline", {}).get("status") == "completed" for row in rows),
    "latency_seconds": {"median": round(statistics.median(latencies), 3),
                        "p95_nearest_rank": round(latencies[math.ceil(len(latencies) * .95) - 1], 3)},
    "tokens_spent_this_experiment": report["after_budget"]["spent_tokens"] - report["before_budget"]["spent_tokens"],
    "cumulative_budget": {**report["after_budget"], "approved_limit": 1_500_000},
    "actual_models": sorted({row["agent"]["model_snapshot"].get("actual_model") or "unknown" for row in completed}),
    "routes": dict(Counter(row["agent"].get("route") for row in completed)),
    "retrieve_retry_runs": sum(bool(row["agent"].get("retry_retrieve")) for row in completed),
    "generate_retry_runs": sum(bool(row["agent"].get("retry_generate")) for row in completed),
    "interpretation": "Completed includes clarification/refusal. Verified is a model flag, not correctness. v1 predates truncation guard and shorter prompt; no improvement claim.",
}
selection = json.loads((ROOT / "output/kg_pilot/selection_300.json").read_text(encoding="utf-8"))
with sqlite3.connect((ROOT / "output/kg_pilot/checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
    entries = db.execute("SELECT chunk_id,status,triples_json FROM chunks WHERE snapshot_id=?", (selection["snapshot_id"],)).fetchall()
statuses = {chunk: status for chunk, status, _ in entries}
done = sum(all(statuses.get(chunk) in {"completed", "no_relations"} for chunk in chunks)
           for chunks in selection["chunk_ids"].values())
summary["kg_pilot"] = {"selected_documents": len(selection["documents"]), "complete_documents": done,
    "partial_documents": sum(any(chunk in statuses for chunk in chunks) and
                            not all(statuses.get(chunk) in {"completed", "no_relations"} for chunk in chunks)
                            for chunks in selection["chunk_ids"].values()),
    "chunk_statuses": dict(Counter(status for _, status, _ in entries)),
    "extracted_triples": sum(len(json.loads(payload)) for _, status, payload in entries if status == "completed"),
    "semantic_acceptance": "not_approved", "online_default": "off"}
destination = ROOT / "output/quality/delivery-eval-summary-20260930.json"
destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
