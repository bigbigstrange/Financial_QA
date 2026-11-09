"""Authoritative document/version/snapshot catalog with guarded transitions."""
from __future__ import annotations

import json
from typing import Iterable

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeSnapshotRecord,
)
from src.contracts import utc_now
from src.pipeline.identity import ParseContext, make_document_version


ALLOWED_VERSION_TRANSITIONS = {
    "staged": {"validating", "failed"},
    "validating": {"ready", "failed"},
    "ready": {"active", "deleted"},
    "active": {"superseded", "deleted"},
    "superseded": {"active", "deleted"},
    "failed": set(),
    "deleted": set(),
}


class DocumentCatalogError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DocumentCatalog:
    def __init__(self, session: Session) -> None:
        self.session = session

    def stage_version(
        self,
        *,
        document_id: str,
        source_sha256: str,
        source_size_bytes: int,
        source_format: str,
        parser_name: str,
        parser_version: str,
        parser_config_hash: str,
        chunker_version: str,
        docir_path: str | None = None,
        quality_report_path: str | None = None,
    ) -> DocumentVersionRecord:
        existing = self._find_version(
            document_id,
            source_sha256,
            parser_config_hash,
            chunker_version,
        )
        if existing is not None:
            return existing

        context = ParseContext(
            document_id=document_id,
            source_name="",
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            parser_config_hash=parser_config_hash,
            chunker_version=chunker_version,
        )
        version_id = make_document_version(context, parser_name, parser_version)
        record = DocumentVersionRecord(
            id=version_id,
            document_id=document_id,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            source_format=source_format,
            parser_name=parser_name,
            parser_version=parser_version,
            parser_config_hash=parser_config_hash,
            chunker_version=chunker_version,
            status="staged",
            docir_path=docir_path,
            quality_report_path=quality_report_path,
        )
        self.session.add(record)
        try:
            self.session.flush()
            return record
        except IntegrityError:
            self.session.rollback()
            existing = self._find_version(
                document_id,
                source_sha256,
                parser_config_hash,
                chunker_version,
            )
            if existing is not None:
                return existing
            raise

    def mark_ready(self, version_id: str) -> DocumentVersionRecord:
        record = self.get_version(version_id)
        if record.status == "ready":
            return record
        if record.status == "staged":
            self._transition(record, "validating")
        self._transition(record, "ready")
        return record

    def mark_failed(
        self,
        version_id: str,
        *,
        code: str,
        message: str,
    ) -> DocumentVersionRecord:
        record = self.get_version(version_id)
        self._transition(record, "failed")
        record.failure_code = code
        record.failure_message = message
        return record

    def activate(
        self,
        version_id: str,
        *,
        snapshot_id: str,
        manifest_path: str,
        block_count: int,
        manifest_sha256: str,
        versions: Iterable[str],
    ) -> KnowledgeSnapshotRecord:
        version = self.get_version(version_id)
        if version.status not in {"ready", "superseded", "active"}:
            raise DocumentCatalogError(
                "invalid_transition",
                f"cannot activate version in {version.status}",
            )
        document = self.session.get(DocumentRecord, version.document_id)
        if document is None or document.status == "deleted":
            raise DocumentCatalogError("document_unavailable", "文档不可用")

        previous = (
            self.session.get(DocumentVersionRecord, document.active_version_id)
            if document.active_version_id
            else None
        )
        if previous is not None and previous.id != version.id:
            self._transition(previous, "superseded")

        version.status = "active"
        version.activated_at = utc_now()
        document.active_version_id = version.id
        document.status = "active"
        document.updated_at = utc_now()

        for snapshot in (
            self.session.query(KnowledgeSnapshotRecord)
            .filter(KnowledgeSnapshotRecord.status == "active")
            .all()
        ):
            snapshot.status = "retired"

        snapshot = self.session.get(KnowledgeSnapshotRecord, snapshot_id)
        version_list = sorted(set(versions))
        if snapshot is None:
            snapshot = KnowledgeSnapshotRecord(
                id=snapshot_id,
                manifest_path=manifest_path,
                status="active",
                document_versions_json=json.dumps(version_list, ensure_ascii=False),
                parser_schema_version=2,
                chunker_version=version.chunker_version,
                block_count=block_count,
                manifest_sha256=manifest_sha256,
                activated_at=utc_now(),
            )
            self.session.add(snapshot)
        else:
            snapshot.status = "active"
            snapshot.manifest_path = manifest_path
            snapshot.document_versions_json = json.dumps(
                version_list, ensure_ascii=False
            )
            snapshot.block_count = block_count
            snapshot.manifest_sha256 = manifest_sha256
            snapshot.activated_at = utc_now()
        self.session.flush()
        return snapshot

    def delete_document(self, document_id: str) -> list[str]:
        document = self.get_document(document_id)
        document.status = "deleted"
        document.deleted_at = utc_now()
        document.active_version_id = None
        versions = (
            self.session.query(DocumentVersionRecord)
            .filter(
                DocumentVersionRecord.document_id == document_id,
                DocumentVersionRecord.status != "deleted",
            )
            .all()
        )
        deleted: list[str] = []
        for version in versions:
            self._transition(version, "deleted")
            deleted.append(version.id)
        self.session.flush()
        return deleted

    def rollback(
        self,
        document_id: str,
        version_id: str,
    ) -> DocumentVersionRecord:
        document = self.get_document(document_id)
        version = self.get_version(version_id)
        if version.status not in {"ready", "superseded", "active"}:
            raise DocumentCatalogError(
                "invalid_rollback",
                f"cannot rollback to version in {version.status}",
            )
        previous = (
            self.session.get(DocumentVersionRecord, document.active_version_id)
            if document.active_version_id
            else None
        )
        if previous is not None and previous.id != version.id:
            self._transition(previous, "superseded")
        version.status = "active"
        version.activated_at = utc_now()
        document.active_version_id = version.id
        document.status = "active"
        document.updated_at = utc_now()
        self.session.flush()
        return version

    def active_version_ids(self) -> frozenset[str]:
        rows = (
            self.session.query(DocumentVersionRecord.id)
            .filter(DocumentVersionRecord.status == "active")
            .all()
        )
        return frozenset(row[0] for row in rows)

    def active_snapshot(self) -> KnowledgeSnapshotRecord | None:
        return (
            self.session.query(KnowledgeSnapshotRecord)
            .filter(KnowledgeSnapshotRecord.status == "active")
            .first()
        )

    def get_document(self, document_id: str) -> DocumentRecord:
        document = self.session.get(DocumentRecord, document_id)
        if document is None:
            raise DocumentCatalogError("document_not_found", "文档不存在")
        return document

    def get_version(self, version_id: str) -> DocumentVersionRecord:
        version = self.session.get(DocumentVersionRecord, version_id)
        if version is None:
            raise DocumentCatalogError("version_not_found", "版本不存在")
        return version

    def _find_version(
        self,
        document_id: str,
        source_sha256: str,
        parser_config_hash: str,
        chunker_version: str,
    ) -> DocumentVersionRecord | None:
        return (
            self.session.query(DocumentVersionRecord)
            .filter(
                DocumentVersionRecord.document_id == document_id,
                DocumentVersionRecord.source_sha256 == source_sha256,
                DocumentVersionRecord.parser_config_hash == parser_config_hash,
                DocumentVersionRecord.chunker_version == chunker_version,
            )
            .first()
        )

    def _transition(
        self,
        record: DocumentVersionRecord,
        target: str,
    ) -> None:
        if target not in ALLOWED_VERSION_TRANSITIONS.get(record.status, set()):
            raise DocumentCatalogError(
                "invalid_transition",
                f"cannot move {record.status} -> {target}",
            )
        record.status = target
