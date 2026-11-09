"""Initialize a fresh catalogue once; retain an existing active snapshot."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.db.init_db import init_db
from app.db.session import SessionLocal
from app.models import KnowledgeSnapshotRecord

init_db()
with SessionLocal() as db:
    active = db.query(KnowledgeSnapshotRecord).filter_by(status="active").first()
if active is None:
    subprocess.run([sys.executable, str(ROOT / "scripts/register_kb_snapshot.py"),
                    "--manifest", get_settings().kb_path, "--activate"], check=True)
os.execv(sys.executable, [sys.executable, "-m", "app.main"])
