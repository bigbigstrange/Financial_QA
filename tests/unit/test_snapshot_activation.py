import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import (
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeSnapshotRecord,
)
from app.services.document_catalog import DocumentCatalog
from app.services.document_service import DocumentService, DocumentServiceError
from src.pipeline.docir import DocBlock, DocIR, DocPage
from src.pipeline.snapshots import (
    SnapshotConfig,
    SnapshotManifest,
    VersionArtifact,
    read_manifest,
    write_manifest_atomic,
)


class SnapshotActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_manifest_id_is_order_independent(self):
        a = VersionArtifact("doc-a", "dv-a", "a.docir.json", 1, "sha-a")
        b = VersionArtifact("doc-b", "dv-b", "b.docir.json", 1, "sha-b")
        config = SnapshotConfig(2, "chunk-v2", "parsers-v2")

        one = SnapshotManifest.build([b, a], config)
        two = SnapshotManifest.build([a, b], config)

        self.assertEqual(one.snapshot_id, two.snapshot_id)
        self.assertTrue(one.manifest_sha256)

    def test_manifest_round_trip_and_hash_mismatch(self):
        artifact = VersionArtifact("doc-a", "dv-a", "a.docir.json", 1, "sha-a")
        config = SnapshotConfig(2, "chunk-v2", "parsers-v2")
        manifest = SnapshotManifest.build([artifact], config)
        path = self.tmp_path / "snapshot.json"
        write_manifest_atomic(manifest, path)

        restored = read_manifest(path)
        self.assertEqual(restored.snapshot_id, manifest.snapshot_id)

        data = json.loads(path.read_text(encoding="utf-8"))
        data["manifest_sha256"] = "tampered"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_manifest(path)

    def _write_docir(self, path: Path, version: str) -> DocIR:
        doc = DocIR(
            doc_id="doc-new",
            source_path="new.txt",
            source_format=".txt",
            document_version=version,
            schema_version=2,
        )
        doc.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock(
                        "blk-new",
                        "paragraph",
                        "正文",
                        None,
                        structural_locator="body/000000/paragraph",
                    )
                ],
            )
        )
        path.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        return doc

    def test_activation_failure_keeps_previous_snapshot(self):
        document = DocumentRecord(id="doc-new", display_name="new.txt", status="active")
        docir_path = self.tmp_path / "new.docir.json"
        self._write_docir(docir_path, "dv-new")
        version = DocumentVersionRecord(
            id="dv-new",
            document_id="doc-new",
            source_sha256="new",
            source_size_bytes=1,
            source_format=".txt",
            parser_name="plain_text",
            parser_version="2",
            parser_config_hash="cfg",
            chunker_version="chunk-v2",
            status="ready",
            docir_path=str(docir_path),
            quality_report_path="quality.json",
        )
        old = KnowledgeSnapshotRecord(
            id="snap-old",
            manifest_path="old.json",
            status="active",
            document_versions_json="[]",
            parser_schema_version=2,
            chunker_version="chunk-v2",
            block_count=0,
            manifest_sha256="old",
        )
        self.session.add_all([document, version, old])
        self.session.commit()

        service = DocumentService(
            self.session,
            artifact_root=self.tmp_path,
            catalog=DocumentCatalog(self.session),
        )
        with patch(
            "app.services.document_service.write_manifest_atomic",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(DocumentServiceError) as raised:
                service.activate_ready_version("dv-new")
        self.assertEqual(raised.exception.code, "snapshot_write_failed")
        self.session.rollback()
        self.assertEqual(
            DocumentCatalog(self.session).active_snapshot().id,
            "snap-old",
        )


if __name__ == "__main__":
    unittest.main()
