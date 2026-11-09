import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models import KnowledgeSnapshotRecord
from scripts.migrate_phase2_knowledge import (
    MigrationGateError,
    MigrationOptions,
    run_migration,
)
from src.pipeline.quality import SnapshotQualityReport


class Phase2MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.input_dir = self.tmp_path / "corpus"
        self.input_dir.mkdir()
        (self.input_dir / "rules.txt").write_text("规则正文", encoding="utf-8")
        (self.input_dir / "faq.txt").write_text("常见问题", encoding="utf-8")
        self.database_path = self.tmp_path / "migration.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _seed_old_snapshot(self) -> None:
        engine = create_engine(f"sqlite:///{self.database_path}", future=True)
        Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine, future=True)()
        session.add(
            KnowledgeSnapshotRecord(
                id="snap-old",
                manifest_path="old.json",
                status="active",
                document_versions_json="[]",
                parser_schema_version=2,
                chunker_version="chunk-v2",
                block_count=0,
                manifest_sha256="old",
            )
        )
        session.commit()
        session.close()
        engine.dispose()

    def test_migration_default_never_switches_active_snapshot(self):
        self._seed_old_snapshot()

        result = run_migration(
            MigrationOptions(
                input_dir=str(self.input_dir),
                candidate_dir=str(self.tmp_path / "candidate"),
                database_url=f"sqlite:///{self.database_path}",
                activate=False,
            )
        )

        self.assertTrue(result.candidate_snapshot_id)
        self.assertEqual(result.document_count, 2)
        self.assertFalse(result.activated)
        engine = create_engine(f"sqlite:///{self.database_path}", future=True)
        session = sessionmaker(bind=engine, future=True)()
        active = (
            session.query(KnowledgeSnapshotRecord)
            .filter(KnowledgeSnapshotRecord.status == "active")
            .first()
        )
        self.assertEqual(active.id, "snap-old")
        session.close()
        engine.dispose()

    def test_failed_quality_gate_rejects_activate(self):
        failed = SnapshotQualityReport(
            snapshot_id="snap-x",
            document_count=0,
            block_count=0,
            duplicate_block_id_count=0,
            missing_locator_count=0,
            empty_block_count=0,
            table_count=0,
            table_row_group_count=0,
            hard_failure_count=1,
            status="failed",
            failure_reasons=["bad"],
        )
        with patch(
            "scripts.migrate_phase2_knowledge.assess_snapshot",
            return_value=failed,
        ):
            with self.assertRaises(MigrationGateError) as raised:
                run_migration(
                    MigrationOptions(
                        input_dir=str(self.input_dir),
                        candidate_dir=str(self.tmp_path / "candidate"),
                        database_url=f"sqlite:///{self.database_path}",
                        activate=True,
                    )
                )
        self.assertEqual(raised.exception.code, "candidate_quality_failed")


if __name__ == "__main__":
    unittest.main()
