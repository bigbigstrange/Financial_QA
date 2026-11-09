import subprocess
import sys
import unittest
from pathlib import Path


class StartupRecoveryTests(unittest.TestCase):
    def test_lifespan_reconciles_interrupted_run(self):
        # app.main loads .env at import time; isolate that process-wide side effect.
        script = '''
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app import main as application
from app.db import init_db as bootstrap
from app.db.base import Base
from app.models import RunRecord, User

engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
sessions = sessionmaker(bind=engine)
with sessions() as db:
    db.add(User(username="interrupted-user", email="interrupted@example.com", password_hash="unused", role="user"))
    db.flush()
    db.add(RunRecord(id="interrupted-run", user_id=1, status="running", question="pending", model_snapshot_json="{}"))
    db.commit()
with patch.object(bootstrap, "engine", engine), patch.object(bootstrap, "SessionLocal", sessions), patch.object(application, "SessionLocal", sessions):
    with TestClient(application.app):
        pass
with sessions() as db:
    record = db.query(RunRecord).filter_by(id="interrupted-run").one()
    assert record.status == "failed"
    assert record.finished_at is not None
engine.dispose()
'''
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
