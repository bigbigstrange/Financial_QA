"""Exact cache compatibility contract for derived vector indexes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.pipeline.identity import stable_digest


@dataclass(frozen=True)
class IndexFingerprint:
    snapshot_id: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    normalization_method: str
    distance_metric: str
    docir_schema_version: int
    chunker_version: str
    parser_versions_hash: str
    retrieval_channel: str
    demo_subset_hash: str | None = None

    def cache_key(self) -> str:
        return stable_digest(json.dumps(asdict(self), sort_keys=True))[:24]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matches(self, metadata: dict[str, Any]) -> bool:
        if not isinstance(metadata, dict):
            return False
        return all(
            metadata.get(key) == value
            for key, value in asdict(self).items()
        )


def fingerprint_meta_path(index_path: str | Path) -> Path:
    return Path(str(index_path) + ".fingerprint.json")


def write_index_fingerprint_meta(
    index_path: str | Path,
    fingerprint: IndexFingerprint,
    *,
    block_ids_sha256: str | None = None,
) -> Path:
    meta_path = fingerprint_meta_path(index_path)
    payload = dict(fingerprint.to_dict())
    payload["block_ids_sha256"] = block_ids_sha256
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return meta_path


def load_index_fingerprint_meta(index_path: str | Path) -> dict[str, Any] | None:
    meta_path = fingerprint_meta_path(index_path)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def index_meta_matches(
    fingerprint: IndexFingerprint,
    metadata: dict[str, Any] | None,
    *,
    block_ids_sha256: str | None = None,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    if not fingerprint.matches(metadata):
        return False
    if block_ids_sha256 is not None and metadata.get("block_ids_sha256") != block_ids_sha256:
        return False
    return True
