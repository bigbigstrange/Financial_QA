import unittest
from sqlalchemy import create_engine, text, inspect
from app.db.base import Base
from app.db.schema_upgrades import run_schema_upgrades
from app.services.run_service import RunService
from sqlalchemy.orm import Session
from src.contracts import Evidence


class SchemaUpgradeTests(unittest.TestCase):
    def test_run_checkpoint_columns_are_additive_and_history_is_unchanged(self):
        engine = create_engine("sqlite://")
        self.addCleanup(engine.dispose)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE runs (id TEXT PRIMARY KEY, answer TEXT)"))
            conn.execute(text("INSERT INTO runs VALUES ('old', 'historical answer')"))
        Base.metadata.create_all(engine)
        run_schema_upgrades(engine)
        run_schema_upgrades(engine)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT answer, checkpoint_json, cancel_requested_at, options_json FROM runs WHERE id='old'")).one()
        self.assertEqual(tuple(row), ("historical answer", None, None, "{}"))
        self.assertIn("run_events", inspect(engine).get_table_names())
        self.assertIn("review_tickets", inspect(engine).get_table_names())

    def test_legacy_evidence_preserved_and_upgrade_idempotent(self):
        engine = create_engine("sqlite://")
        self.addCleanup(engine.dispose)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE evidence_records (id TEXT PRIMARY KEY, document_version TEXT, quote TEXT)"))
            conn.execute(text("INSERT INTO evidence_records VALUES ('ev1','dv1','original')"))
        run_schema_upgrades(engine)
        run_schema_upgrades(engine)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT document_version_id, quote, evidence_json FROM evidence_records")).one()
        self.assertEqual(tuple(row), ("dv1", "original", "{}"))

    def test_new_evidence_can_be_written_to_required_legacy_columns(self):
        engine = create_engine("sqlite://")
        self.addCleanup(engine.dispose)
        with engine.begin() as conn:
            conn.execute(text("""CREATE TABLE evidence_records (
                id TEXT NOT NULL, run_id TEXT NOT NULL, user_id INTEGER NOT NULL,
                snapshot_id TEXT NOT NULL, document_id TEXT NOT NULL,
                document_version TEXT NOT NULL, block_id TEXT NOT NULL,
                quote TEXT NOT NULL, scores_json TEXT NOT NULL,
                PRIMARY KEY (id, run_id))"""))
        run_schema_upgrades(engine)
        ev = Evidence(evidence_id="ev1", document_id="doc1", document_version="dv1", block_id="ck1",
                      text="原文", metadata={"snapshot_id": "snap1"})
        with Session(engine) as db:
            RunService._insert_evidence(db, ev, run_id="run1", user_id=1)
            db.commit()
            row = db.execute(text("SELECT document_version, document_version_id, quote, snapshot_id FROM evidence_records")).one()
        self.assertEqual(tuple(row), ("dv1", "dv1", "原文", "snap1"))
