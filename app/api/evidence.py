"""Authenticated evidence location endpoint."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models import User
from app.schemas import EvidenceDetailOut
from app.services.evidence_reader import (
    EvidenceIntegrityError,
    EvidenceNotFoundError,
    EvidenceReader,
)


router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get('/{evidence_id}/image')
def get_evidence_image(evidence_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    try:
        data, media_type = EvidenceReader.read_image(db, evidence_id, user)
    except EvidenceIntegrityError as exc:
        raise HTTPException(409, detail={'code': 'evidence_identity_mismatch',
                                       'message': '证据完整性校验失败'}) from exc
    except EvidenceNotFoundError as exc:
        raise HTTPException(404, detail={'code': 'evidence_not_found', 'message': '证据不存在'}) from exc
    return Response(data, media_type=media_type,
        headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})


@router.get("/{evidence_id}", response_model=EvidenceDetailOut)
def get_evidence(
    evidence_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> EvidenceDetailOut:
    try:
        detail = EvidenceReader.read(db, evidence_id, user)
    except EvidenceIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "evidence_identity_mismatch", "message": str(exc)},
        ) from exc
    except EvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "evidence_not_found", "message": "证据不存在"},
        ) from exc
    return EvidenceDetailOut.model_validate(detail)
