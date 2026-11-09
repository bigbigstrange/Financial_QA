"""Orchestration: validate -> parse -> quality gate -> snapshot activation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.models import (
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeSnapshotRecord,
)
from app.services.document_catalog import DocumentCatalog
from src.contracts import utc_now
from src.pipeline.file_validation import (
    FileLimits,
    ValidatedFile,
    validate_input_file,
)
from src.pipeline.identity import ParseContext
from src.pipeline.parsers import build_default_parsers
from src.pipeline.quality import process_document
from src.pipeline.snapshots import (
    SnapshotConfig,
    SnapshotManifest,
    VersionArtifact,
    sha256_file,
    write_manifest_atomic,
)


class DocumentServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DocumentService:
    def __init__(
        self,
        session: Session,
        *,
        artifact_root: str | Path,
        catalog: Optional[DocumentCatalog] = None,
        limits: Optional[FileLimits] = None,
    ) -> None:
        self.session = session
        self.artifact_root = Path(artifact_root)
        self.catalog = catalog or DocumentCatalog(session)
        self.limits = limits or FileLimits()

    def ingest(
        self,
        file_path: str | Path,
        *,
        document_id: str,
        display_name: Optional[str] = None,
        owner_id: Optional[int] = None,
    ) -> DocumentVersionRecord:
        path = Path(file_path)
        validated = validate_input_file(path, self.limits)
        document = self.session.get(DocumentRecord, document_id)
        if document is None:
            document = DocumentRecord(
                id=document_id,
                display_name=display_name or path.name,
                owner_id=owner_id,
                status="active",
            )
            self.session.add(document)
            self.session.flush()
        elif display_name:
            document.display_name = display_name

        context = ParseContext.for_file(path, document_id)
        processed = process_document(
            path,
            self.artifact_root / "docir",
            context,
            build_default_parsers(),
        )
        version = self.catalog.stage_version(
            document_id=document_id,
            source_sha256=validated.source_sha256,
            source_size_bytes=validated.size_bytes,
            source_format=validated.detected_format,
            parser_name=processed.document.parser_name or "unknown",
            parser_version=processed.document.parser_version or "1",
            parser_config_hash=context.parser_config_hash,
            chunker_version=context.chunker_version,
            docir_path=str(processed.docir_path),
            quality_report_path=str(processed.quality_path),
        )
        if version.status == "staged":
            version.docir_path = str(processed.docir_path)
            version.quality_report_path = str(processed.quality_path)
            self.catalog.mark_ready(version.id)
        self._activate_snapshot_containing(version.id)
        return version

    def delete(self, document_id: str) -> list[str]:
        deleted = self.catalog.delete_document(document_id)
        self._rebuild_snapshot_after_change()
        return deleted

    def rollback(self, document_id: str, version_id: str) -> DocumentVersionRecord:
        version = self.catalog.rollback(document_id, version_id)
        self._activate_snapshot_containing(version.id)
        return version

    def activate_ready_version(self, version_id: str) -> KnowledgeSnapshotRecord:
        version = self.catalog.get_version(version_id)
        if version.status == "ready":
            self.catalog.mark_ready(version.id)
        return self._activate_snapshot_containing(version.id)

    def status(self, document_id: str) -> dict:
        document = self.catalog.get_document(document_id)
        versions = (
            self.session.query(DocumentVersionRecord)
            .filter(DocumentVersionRecord.document_id == document_id)
            .order_by(DocumentVersionRecord.created_at.desc())
            .all()
        )
        return {
            "document_id": document.id,
            "display_name": document.display_name,
            "status": document.status,
            "active_version_id": document.active_version_id,
            "versions": [
                {
                    "id": version.id,
                    "status": version.status,
                    "source_sha256": version.source_sha256,
                    "parser_name": version.parser_name,
                    "parser_version": version.parser_version,
                }
                for version in versions
            ],
        }

    def _activate_snapshot_containing(
        self,
        version_id: str,
    ) -> KnowledgeSnapshotRecord:
        target = self.catalog.get_version(version_id)
        version_ids = set(self.catalog.active_version_ids())
        if target.status == "active":
            version_ids.add(target.id)
        else:
            version_ids.add(target.id)

        artifacts = []
        for version in (
            self.session.query(DocumentVersionRecord)
            .filter(DocumentVersionRecord.id.in_(sorted(version_ids)))
            .all()
        ):
            docir_path = Path(version.docir_path or "")
            if not docir_path.is_file():
                raise DocumentServiceError(
                    "missing_docir",
                    f"版本缺少 DocIR 文件: {version.id}",
                )
            document = DocIRLoader.load(docir_path)
            artifacts.append(
                VersionArtifact(
                    document_id=document.document_id,
                    document_version=document.document_version,
                    docir_path=str(docir_path),
                    block_count=len(document.all_blocks()),
                    docir_sha256=sha256_file(docir_path),
                )
            )

        config = SnapshotConfig(
            docir_schema_version=2,
            chunker_version=target.chunker_version,
            parser_versions_hash=str(
                sorted(
                    {f"{item.parser_name}:{item.parser_version}" for item in
                     self.session.query(DocumentVersionRecord).filter(
                         DocumentVersionRecord.id.in_(sorted(version_ids))
                     ).all()}
                )
            ),
        )
        manifest = SnapshotManifest.build(artifacts, config)
        manifest_path = (
            self.artifact_root
            / "snapshots"
            / manifest.snapshot_id
            / "manifest.json"
        )
        try:
            write_manifest_atomic(manifest, manifest_path)
        except OSError as exc:
            raise DocumentServiceError(
                "snapshot_write_failed",
                f"无法写入知识快照: {exc}",
            ) from exc
        return self.catalog.activate(
            target.id,
            snapshot_id=manifest.snapshot_id,
            manifest_path=str(manifest_path),
            block_count=manifest.block_count,
            manifest_sha256=manifest.manifest_sha256,
            versions=[artifact.document_version for artifact in artifacts],
        )

    def _rebuild_snapshot_after_change(self) -> None:
        active_versions = (
            self.session.query(DocumentVersionRecord)
            .filter(DocumentVersionRecord.status == "active")
            .all()
        )
        artifacts = []
        for version in active_versions:
            docir_path = Path(version.docir_path or "")
            if not docir_path.is_file():
                continue
            document = DocIRLoader.load(docir_path)
            artifacts.append(
                VersionArtifact(
                    document_id=document.document_id,
                    document_version=document.document_version,
                    docir_path=str(docir_path),
                    block_count=len(document.all_blocks()),
                    docir_sha256=sha256_file(docir_path),
                )
            )
        config = SnapshotConfig(2, "chunk-v2", "parsers-v2")
        manifest = SnapshotManifest.build(artifacts, config)
        manifest_path = (
            self.artifact_root
            / "snapshots"
            / manifest.snapshot_id
            / "manifest.json"
        )
        write_manifest_atomic(manifest, manifest_path)

        for snapshot in (
            self.session.query(KnowledgeSnapshotRecord)
            .filter(KnowledgeSnapshotRecord.status == "active")
            .all()
        ):
            snapshot.status = "retired"
        snapshot = self.session.get(
            KnowledgeSnapshotRecord,
            manifest.snapshot_id,
        )
        versions_json = json.dumps(
            [item.document_version for item in manifest.versions],
            ensure_ascii=False,
        )
        if snapshot is None:
            self.session.add(
                KnowledgeSnapshotRecord(
                    id=manifest.snapshot_id,
                    manifest_path=str(manifest_path),
                    status="active",
                    document_versions_json=versions_json,
                    parser_schema_version=2,
                    chunker_version="chunk-v2",
                    block_count=manifest.block_count,
                    manifest_sha256=manifest.manifest_sha256,
                    activated_at=utc_now(),
                )
            )
        else:
            snapshot.status = "active"
            snapshot.manifest_path = str(manifest_path)
            snapshot.document_versions_json = versions_json
            snapshot.block_count = manifest.block_count
            snapshot.manifest_sha256 = manifest.manifest_sha256
            snapshot.activated_at = utc_now()
        self.session.flush()


class DocIRLoader:
    @staticmethod
    def load(path: Path):
        from src.pipeline.docir import DocIR

        return DocIR.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
