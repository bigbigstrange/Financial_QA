"""Query-time visibility filtering for active document versions."""
from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.models import DocumentVersionRecord


class VersionVisibility(Protocol):
    def allowed(self, version_id: str) -> bool: ...


def filter_visible_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
    visibility: VersionVisibility,
) -> list[dict[str, Any]]:
    allowed = [
        candidate
        for candidate in candidates
        if visibility.allowed(candidate.get("document_version") or "")
    ]
    return allowed[: max(0, int(top_k))]


class CatalogVersionVisibility:
    def __init__(self, session: Session) -> None:
        self.session = session

    def allowed(self, version_id: str) -> bool:
        if not version_id:
            return False
        return (
            self.session.query(DocumentVersionRecord.id)
            .filter(
                DocumentVersionRecord.id == version_id,
                DocumentVersionRecord.status == "active",
            )
            .first()
            is not None
        )
