"""Resume all PDF/image parsing, then build an inactive full-corpus candidate.

Office conversion runs separately and must be finished before this command.
Every business file appears in coverage.json, including failures/duplicates.
The active KB is never switched by this script.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.identity import artifact_key


def build_coverage(records, kb, converted_files, conversion_rows):
    docs = kb["documents"]
    origins = {d.get("origin_rel_path") for d in docs if d.get("origin_rel_path")}
    keys = {d["doc_key"] for d in docs}
    conversions = {r["origin_rel_path"]: r for r in converted_files}
    attempts = {r["rel_path"]: r for r in conversion_rows}
    skipped = {r["doc_key"]: r["reason"] for r in kb.get("stats", {}).get("skipped", [])}
    coverage = []
    for record in records:
        rel = record["rel_path"]
        converted = conversions.get(rel)
        key = artifact_key(converted["rel_path"] if converted else rel)
        row = {"rel_path": rel, "status": "needs_review"}
        if rel in origins or key in keys:
            row["status"] = "indexed"
        elif skipped.get(key, "").startswith("duplicate_of:"):
            canonical = skipped[key].split(":", 1)[1]
            if canonical in keys:
                row.update(status="duplicate", canonical_doc_key=canonical)
            else:
                row["reason"] = "duplicate_target_missing"
        else:
            attempt = attempts.get(rel, {})
            row["reason"] = skipped.get(key) or (
                "conversion_" + attempt["status"] if attempt.get("status") in {"failed", "timeout"}
                else "parse_missing_or_no_retrievable_content")
        coverage.append(row)
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, action="append", default=[],
                        help="Wait for currently running native parsing/conversion before resuming")
    args = parser.parse_args()
    target = ROOT / "output/kb_full"
    inventory = target / "full_inventory.json"
    converted = target / "converted/subset_converted.json"
    records = json.loads(inventory.read_text(encoding="utf-8"))["files"]
    state_path = target / "pipeline_state.json"

    def state(stage, **extra):
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stage": stage, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(state_path)

    def run(stage, arguments):
        state(stage)
        with (target / f"{stage}.log").open("a", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        state(stage + "_finished", exit_code=proc.returncode)
        return proc.returncode

    if args.wait_pid:
        import psutil
        watched = []
        for pid in args.wait_pid:
            try:
                proc = psutil.Process(pid)
                command = " ".join(proc.cmdline())
                if not any(name in command for name in ("run_mineru_batch.py", "convert_office_isolated.py")):
                    raise ValueError(f"PID {pid} is not a corpus processing job")
                watched.append(proc)
            except psutil.NoSuchProcess:
                pass
        state("waiting_for_existing_jobs", pids=[p.pid for p in watched])
        while any(p.is_running() for p in watched):
            time.sleep(10)

    common = ["scripts/run_mineru_batch.py", "--workers", "1", "--timeout", "3600",
              "--shard-pages", "30", "--out-root", str(ROOT / "output/kb_v3/parsed"),
              "--manifest", str(target / "mineru_manifest.jsonl")]
    run("parse_native", common + ["--subset", str(inventory), "--ext", ".pdf", "--ext", ".png", "--ext", ".jpg"])
    if converted.exists():
        run("parse_office", common + ["--subset", str(converted), "--docs-root", str(target / "converted")])
    code = run("build_candidate", ["scripts/build_kb_v3.py", "--output-dir", str(target),
        "--subset", str(inventory), "--subset", str(converted),
        "--docs-root", str(ROOT / "data/documents"), "--docs-root", str(target / "converted")])
    if code:
        state("build_failed", exit_code=code)
        raise SystemExit(code)
    kb = json.loads((target / "kb_v3.json").read_text(encoding="utf-8"))
    docs = kb["documents"]
    converted_files = json.loads(converted.read_text(encoding="utf-8"))["files"]
    conversion_rows = [json.loads(line) for line in (target / "convert_isolated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    coverage = build_coverage(records, kb, converted_files, conversion_rows)
    (target / "coverage.json").write_text(json.dumps({"expected_files": len(records),
        "indexed_documents": len(docs), "files": coverage}, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = sum(r["status"] == "needs_review" for r in coverage)
    state("candidate_ready_for_review", expected_files=len(records), indexed_documents=len(docs),
          files_needing_review=missing, activated=False)


if __name__ == "__main__":
    main()
