"""Authorized exact lookup of immutable evidence by identity."""
from __future__ import annotations

import json
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import DocumentVersionRecord, EvidenceRecord, KnowledgeSnapshotRecord, User
from src.pipeline.docir import DocIR
from src.pipeline.snapshots import read_manifest
from src.system.visual_pages import make_visual_page_id, resolve_artifact_path


class EvidenceNotFoundError(RuntimeError):
    pass


class EvidenceIntegrityError(RuntimeError):
    pass


class EvidenceReader:
    @staticmethod
    def _authorized(db, evidence_id, requesting_user):
        record = db.get(EvidenceRecord, evidence_id)
        if record is None or (requesting_user.role != 'admin' and record.user_id != requesting_user.id):
            raise EvidenceNotFoundError('evidence not found')
        return record

    @staticmethod
    def _image_context(db, record):
        try:
            saved = json.loads(record.evidence_json)
            meta = saved.get('metadata') or {}
            snapshot_id = meta.get('snapshot_id')
            snapshot = db.get(KnowledgeSnapshotRecord, snapshot_id) if snapshot_id else None
            if snapshot is None or not meta.get('image_sha256'):
                raise ValueError('missing image identity')
            manifest_path = Path(snapshot.manifest_path)
            manifest = read_manifest(manifest_path)
            if manifest.snapshot_id != snapshot_id or manifest.manifest_sha256 != snapshot.manifest_sha256:
                raise ValueError('snapshot mismatch')
            matches = [a for a in manifest.versions if a.document_version == record.document_version_id
                       and a.document_id == record.document_id]
            if len(matches) != 1:
                raise ValueError('version mismatch')
            roots = (manifest_path.parent, Path(__file__).resolve().parents[2]/'output/kb_v3/parsed')
            path = resolve_artifact_path(matches[0].docir_path, manifest_path.parent, roots)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != matches[0].docir_sha256:
                raise ValueError('docir mismatch')
            document = DocIR.from_dict(json.loads(data))
            if document.document_id != record.document_id or document.document_version != record.document_version_id:
                raise ValueError('document mismatch')
            page_no = meta.get('vision_page')
            page_id = make_visual_page_id(snapshot_id, record.document_id, record.document_version_id, page_no)
            if meta.get('page_id') != page_id:
                raise ValueError('page mismatch')
            if meta.get('evidence_kind') == 'page_image':
                if record.block_id != page_id or saved.get('text') or meta.get('chunk_id') or meta.get('block_ids'):
                    raise ValueError('page evidence mismatch')
            else:
                # Establish the saved chunk before permitting its hit page.
                EvidenceReader._read_chunk(db, record, document)
                with manifest_path.with_name('chunks.jsonl').open(encoding='utf-8') as source:
                    row = next(json.loads(line) for line in source if json.loads(line).get('chunk_id') == record.block_id)
                start = row.get('page_start') or row.get('page')
                end = row.get('page_end') or start
                if start is None or not start <= page_no <= end:
                    raise ValueError('hit outside chunk')
            pages = [p for p in document.pages if p.page_no == page_no]
            if len(pages) != 1:
                raise ValueError('missing or duplicate page')
            reference = pages[0].meta.get('image_ref')
            if not reference:
                raise EvidenceNotFoundError('page image missing')
            image_path = resolve_artifact_path(reference, manifest_path.parent, roots)
            return document, meta, image_path
        except FileNotFoundError as exc:
            raise EvidenceNotFoundError('artifact missing') from exc
        except (ValueError, TypeError, KeyError, StopIteration, OSError) as exc:
            raise EvidenceIntegrityError('evidence_identity_mismatch') from exc

    @staticmethod
    def read_image(db, evidence_id, requesting_user) -> tuple[bytes, str]:
        record = EvidenceReader._authorized(db, evidence_id, requesting_user)
        _, meta, path = EvidenceReader._image_context(db, record)
        try:
            from PIL import Image
            with path.open('rb') as source:
                data = source.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != meta['image_sha256']:
                raise ValueError('image digest mismatch')
            with Image.open(BytesIO(data)) as image:
                media_type = {'PNG': 'image/png', 'JPEG': 'image/jpeg'}.get(image.format)
                if media_type is None:
                    raise ValueError('unsupported image')
                image.verify()
            return data, media_type
        except FileNotFoundError as exc:
            raise EvidenceNotFoundError('page image missing') from exc
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise EvidenceIntegrityError('evidence_identity_mismatch') from exc

    @staticmethod
    def read(
        db: Session,
        evidence_id: str,
        requesting_user: User,
    ) -> dict[str, Any]:
        record = EvidenceReader._authorized(db, evidence_id, requesting_user)
        saved = json.loads(record.evidence_json)
        meta = saved.get('metadata') or {}
        if meta.get('source_url'):
            from src.system.web_search import _public_url
            source_url = meta['source_url']
            if _public_url(source_url) != source_url:
                raise EvidenceIntegrityError('evidence_identity_mismatch')
            return {
                'evidence_id': record.id,
                'evidence_kind': 'web',
                'source_url': source_url,
                'source_name': saved.get('source_name'),
                'document_id': None,
                'document_version': None,
                'block_id': None,
                'page_no': None,
                'text': saved.get('text') or '',
                'quality_flags': saved.get('quality_flags') or [],
            }
        if meta.get('page_id'):
            document, meta, image_path = EvidenceReader._image_context(db, record)
            if meta.get('evidence_kind') == 'page_image':
                detail = dict(evidence_id=record.id, document_id=document.document_id,
                    document_version=document.document_version, block_id=record.block_id,
                    source_name=document.source_name or document.doc_id,
                    page_no=meta['vision_page'], text='', quality_flags=saved.get('quality_flags') or [])
            else:
                detail = EvidenceReader._read_chunk(db, record, document)
            detail.update(evidence_kind=meta.get('evidence_kind', 'text'),
                          vision_page=meta['vision_page'], image_available=image_path.is_file())
            return detail
        if not record.document_version_id or not record.block_id:
            raise EvidenceIntegrityError("evidence_identity_mismatch")

        version = db.get(DocumentVersionRecord, record.document_version_id)
        if version is None:
            raise EvidenceNotFoundError("document version not found")
        if not version.docir_path:
            raise EvidenceIntegrityError("evidence_identity_mismatch")
        docir_path = Path(version.docir_path)
        if not docir_path.is_file():
            raise EvidenceNotFoundError("docir artifact missing")

        document = DocIR.from_dict(
            json.loads(docir_path.read_text(encoding="utf-8"))
        )
        if document.document_version != record.document_version_id:
            raise EvidenceIntegrityError("evidence_identity_mismatch")
        try:
            block = document.require_block(record.block_id)
        except KeyError as exc:
            return EvidenceReader._read_chunk(db, record, document)
        if block.block_id != record.block_id:
            raise EvidenceIntegrityError("evidence_identity_mismatch")

        return {
            "evidence_id": record.id,
            "document_id": document.document_id,
            "document_version": document.document_version,
            "block_id": block.block_id,
            "source_name": document.source_name or document.doc_id,
            "page_no": block.page,
            "bbox": block.bbox,
            "section_path": block.section_path,
            "structural_locator": block.structural_locator,
            "text": block.text,
            "table_headers": block.table_headers,
            "table_rows": block.table_rows,
            "unit": block.unit,
            "quality_flags": block.quality_flags,
        }

    @staticmethod
    def _read_chunk(db, record, document):
        """Resolve v3 chunk identity through its recorded snapshot, never by page guess."""
        saved = json.loads(record.evidence_json)
        snapshot_id = (saved.get("metadata") or {}).get("snapshot_id")
        snapshot = db.get(KnowledgeSnapshotRecord, snapshot_id) if snapshot_id else None
        if snapshot is None:
            raise EvidenceIntegrityError("evidence_identity_mismatch")
        path = Path(snapshot.manifest_path)
        manifest = read_manifest(path)
        if manifest.snapshot_id != snapshot_id or not any(
            a.document_version == record.document_version_id for a in manifest.versions
        ):
            raise EvidenceIntegrityError("evidence_identity_mismatch")
        chunks = path.with_name("chunks.jsonl")
        if not chunks.is_file():
            raise EvidenceIntegrityError("evidence_identity_mismatch")
        with chunks.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("chunk_id") != record.block_id:
                    continue
                if row.get("document_version") != record.document_version_id or row.get("snapshot_id") != snapshot_id:
                    raise EvidenceIntegrityError("evidence_identity_mismatch")
                block_ids = row.get("block_ids") or []
                if not block_ids or row.get("content") != saved.get("text"):
                    raise EvidenceIntegrityError("evidence_identity_mismatch")
                try:
                    originals = [document.require_block(key) for key in block_ids]
                except KeyError as exc:
                    raise EvidenceIntegrityError("evidence_identity_mismatch") from exc
                return {"evidence_id": record.id, "document_id": document.document_id,
                    "document_version": document.document_version, "block_id": record.block_id,
                    "source_name": document.source_name or document.doc_id,
                    "page_no": row.get("page_start", row.get("page")), "bbox": row.get("bbox"),
                    "page_end": row.get("page_end"),
                    "section_path": row.get("section_path"), "structural_locator": None,
                    "text": row["content"], "table_headers": originals[0].table_headers,
                    "table_rows": originals[0].table_rows, "unit": originals[0].unit,
                    "quality_flags": row.get("quality_flags") or []}
        raise EvidenceIntegrityError("evidence_identity_mismatch")
