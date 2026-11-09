"""Online SQLite backup and isolated restore rehearsal; never overwrite live DBs."""
import hashlib
import json
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "output/backups" / ("delivery-" + str(time.time_ns()))
target.mkdir(parents=True)
report = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          "secrets_copied": False, "databases": [], "artifacts": []}
for relative in ("data_app.db", "output/run_checkpoints.sqlite3", "output/pilot_budget.sqlite3"):
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(relative)
    saved = target / source.name
    with sqlite3.connect(source) as db, sqlite3.connect(saved) as backup:
        db.backup(backup)
    restored = target / ("restored-" + source.name)
    with sqlite3.connect(saved) as db, sqlite3.connect(restored) as restore:
        db.backup(restore)
        assert restore.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        counts = {}
        for table in tables:
            name = '"' + table.replace('"', '""') + '"'
            counts[table] = db.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
            assert restore.execute("SELECT COUNT(*) FROM " + name).fetchone()[0] == counts[table]
    report["databases"].append({"source": relative, "backup": saved.name,
        "sha256": hashlib.sha256(saved.read_bytes()).hexdigest(), "restore_integrity": "ok", "row_counts": counts})
for relative in ("configs/model_registry.json", "output/kb_full/kb_v3_manifest.json"):
    source = ROOT / relative
    shutil.copy2(source, target / source.name)
    report["artifacts"].append({"source": relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
(target / "rehearsal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"backup_dir": str(target), "restored_database_count": len(report["databases"]),
                  "status": "passed", "secrets_copied": False}, ensure_ascii=False))
