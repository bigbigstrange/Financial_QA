import unittest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.schema_upgrades import run_schema_upgrades
from app.models import (
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeSnapshotRecord,
)
from app.services.document_catalog import DocumentCatalog, DocumentCatalogError


class DocumentCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )

        @event.listens_for(self.engine, "connect")
        def _fk_on(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(bind=self.engine)
        run_schema_upgrades(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()

    def _document(self, document_id="doc-1", name="rules.pdf") -> DocumentRecord:
        document = DocumentRecord(
            id=document_id,
            display_name=name,
            status="active",
        )
        self.session.add(document)
        return document

    def _version(
        self,
        document_id="doc-1",
        version_id="dv-1",
        sha="a",
        status="ready",
    ) -> DocumentVersionRecord:
        version = DocumentVersionRecord(
            id=version_id,
            document_id=document_id,
            source_sha256=sha,
            source_size_bytes=1,
            source_format=".pdf",
            parser_name="pdf_multimodal",
            parser_version="2",
            parser_config_hash="c",
            chunker_version="chunk-v2",
            status=status,
            docir_path="dv-1.docir.json",
            quality_report_path="dv-1.quality.json",
        )
        self.session.add(version)
        return version

    def test_same_content_and_config_returns_existing_version(self):
        self._document()
        catalog = DocumentCatalog(self.session)
        first = catalog.stage_version(
            document_id="doc-1",
            source_sha256="a",
            source_size_bytes=1,
            source_format=".pdf",
            parser_name="pdf_multimodal",
            parser_version="2",
            parser_config_hash="c",
            chunker_version="chunk-v2",
        )
        second = catalog.stage_version(
            document_id="doc-1",
            source_sha256="a",
            source_size_bytes=1,
            source_format=".pdf",
            parser_name="pdf_multimodal",
            parser_version="2",
            parser_config_hash="c",
            chunker_version="chunk-v2",
        )

        self.assertEqual(first.id, second.id)
        self.assertEqual(
            self.session.query(DocumentVersionRecord)
            .filter(DocumentVersionRecord.document_id == "doc-1")
            .count(),
            1,
        )

    def test_staged_cannot_activate_directly(self):
        document = self._document()
        version = self._version(status="staged")
        document.active_version_id = version.id
        self.session.commit()

        with self.assertRaises(DocumentCatalogError) as raised:
            DocumentCatalog(self.session).activate(
                "dv-1",
                snapshot_id="snap-1",
                manifest_path="snapshot.json",
                block_count=1,
                manifest_sha256="sha",
                versions=["dv-1"],
            )
        self.assertEqual(raised.exception.code, "invalid_transition")

    def test_activate_supersedes_old_version_and_snapshot(self):
        self._document()
        self._version(version_id="dv-old", sha="old", status="active")
        self._version(version_id="dv-new", sha="new", status="ready")
        old_snapshot = KnowledgeSnapshotRecord(
            id="snap-old",
            manifest_path="old.json",
            status="active",
            document_versions_json="[]",
            parser_schema_version=2,
            chunker_version="chunk-v2",
            block_count=0,
            manifest_sha256="old",
        )
        self.session.add(old_snapshot)
        self.session.commit()
        self.session.refresh(self.session.get(DocumentRecord, "doc-1"))
        document = self.session.get(DocumentRecord, "doc-1")
        document.active_version_id = "dv-old"
        self.session.commit()

        DocumentCatalog(self.session).activate(
            "dv-new",
            snapshot_id="snap-new",
            manifest_path="new.json",
            block_count=2,
            manifest_sha256="new",
            versions=["dv-new"],
        )
        self.session.commit()

        document = self.session.get(DocumentRecord, "doc-1")
        self.assertEqual(document.active_version_id, "dv-new")
        self.assertEqual(
            self.session.get(DocumentVersionRecord, "dv-old").status,
            "superseded",
        )
        self.assertEqual(
            self.session.get(KnowledgeSnapshotRecord, "snap-old").status,
            "retired",
        )
        self.assertEqual(DocumentCatalog(self.session).active_snapshot().id, "snap-new")

    def test_delete_hides_document_in_same_transaction(self):
        document = self._document()
        version = self._version(status="active")
        document.active_version_id = version.id
        self.session.commit()

        deleted = DocumentCatalog(self.session).delete_document("doc-1")
        self.session.commit()

        self.assertEqual(deleted, ["dv-1"])
        self.assertEqual(DocumentCatalog(self.session).active_version_ids(), frozenset())
        self.assertEqual(
            self.session.get(DocumentRecord, "doc-1").status,
            "deleted",
        )

    def test_rollback_rejects_non_ready_version(self):
        self._document()
        self._version(version_id="dv-active", sha="active", status="active")
        self._version(version_id="dv-staged", sha="staged", status="staged")
        self.session.commit()
        self.session.get(DocumentRecord, "doc-1").active_version_id = "dv-active"
        self.session.commit()

        with self.assertRaises(DocumentCatalogError) as raised:
            DocumentCatalog(self.session).rollback("doc-1", "dv-staged")
        self.assertEqual(raised.exception.code, "invalid_rollback")


if __name__ == "__main__":
    unittest.main()
