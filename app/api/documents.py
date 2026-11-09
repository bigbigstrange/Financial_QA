"""Admin document intake, status, deletion, and rollback endpoints."""
from __future__ import annotations

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import require_admin
from app.db.session import get_db
from app.models import User
from app.schemas import DocumentIngestResult, DocumentOut
from app.services.document_catalog import DocumentCatalogError
from app.services.document_service import (
    DocumentService,
    DocumentServiceError,
)
from src.pipeline.file_validation import FileValidationError
from src.pipeline.quality import QualityGateError


router = APIRouter(prefix="/documents", tags=["documents"])


def _service(db: Session) -> DocumentService:
    settings = get_settings()
    return DocumentService(db, artifact_root=settings.doc_artifact_root)


@router.post("", response_model=DocumentIngestResult)
def create_document(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
) -> DocumentIngestResult:
    return _ingest(db, file, document_id=str(uuid.uuid4()))


@router.post("/{document_id}/versions", response_model=DocumentIngestResult)
def replace_document(
    document_id: str,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
) -> DocumentIngestResult:
    return _ingest(db, file, document_id=document_id)


@router.get("/{document_id}", response_model=DocumentOut)
def document_status(
    document_id: str,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentOut:
    try:
        return DocumentOut.model_validate(_service(db).status(document_id))
    except DocumentCatalogError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": exc.message}) from exc


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    try:
        _service(db).delete(document_id)
        db.commit()
    except DocumentCatalogError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": exc.message}) from exc
    return {"ok": True}


@router.post("/{document_id}/rollback/{version_id}", response_model=DocumentOut)
def rollback_document(
    document_id: str,
    version_id: str,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentOut:
    try:
        _service(db).rollback(document_id, version_id)
        db.commit()
        return DocumentOut.model_validate(_service(db).status(document_id))
    except DocumentCatalogError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc
    except DocumentServiceError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc


def _ingest(
    db: Session,
    upload: UploadFile,
    *,
    document_id: str,
) -> DocumentIngestResult:
    with TemporaryDirectory() as temp_dir:
        suffix = Path(upload.filename or "upload").suffix
        path = Path(temp_dir) / f"upload-{uuid.uuid4().hex}{suffix}"
        try:
            with path.open("wb") as target:
                while chunk := upload.file.read(1024 * 1024):
                    target.write(chunk)
            service = _service(db)
            version = service.ingest(
                path,
                document_id=document_id,
                display_name=upload.filename,
            )
            db.commit()
            status = service.status(document_id)
            return DocumentIngestResult(
                document_id=document_id,
                display_name=status["display_name"],
                status=status["status"],
                active_version_id=status["active_version_id"],
                created_version_id=version.id,
            )
        except FileValidationError as exc:
            raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc
        except (DocumentServiceError, DocumentCatalogError, QualityGateError) as exc:
            code = getattr(exc, "code", "quality_gate")
            message = str(exc)
            raise HTTPException(status_code=400, detail={"code": code, "message": message}) from exc
