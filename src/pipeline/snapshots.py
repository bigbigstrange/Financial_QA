"""Canonical immutable knowledge snapshot manifests."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .identity import stable_digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class VersionArtifact:
    document_id: str
    document_version: str
    docir_path: str
    block_count: int
    docir_sha256: str


@dataclass
class SnapshotConfig:
    docir_schema_version: int
    chunker_version: str
    parser_versions_hash: str


@dataclass
class SnapshotManifest:
    snapshot_id: str
    config: SnapshotConfig
    versions: list[VersionArtifact]
    block_count: int
    manifest_sha256: str
    schema_version: int = 2

    @classmethod
    def build(
        cls,
        version_artifacts: Iterable[VersionArtifact],
        config: SnapshotConfig,
    ) -> "SnapshotManifest":
        versions = sorted(
            list(version_artifacts),
            key=lambda item: (item.document_version, item.document_id),
        )
        identity = {
            "config": asdict(config),
            "versions": [asdict(item) for item in versions],
            "schema_version": 2,
        }
        snapshot_id = "snap_" + stable_digest(
            json.dumps(identity, ensure_ascii=False, sort_keys=True)
        )[:32]
        manifest = cls(
            snapshot_id=snapshot_id,
            config=config,
            versions=versions,
            block_count=sum(item.block_count for item in versions),
            manifest_sha256=""
        )
        canonical = json.dumps(
            manifest.to_dict(exclude_sha=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest.manifest_sha256 = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return manifest

    def to_dict(self, *, exclude_sha: bool = False) -> dict:
        payload = {
            "snapshot_id": self.snapshot_id,
            "schema_version": self.schema_version,
            "config": asdict(self.config),
            "versions": [asdict(item) for item in self.versions],
            "block_count": self.block_count,
        }
        if not exclude_sha:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotManifest":
        config = SnapshotConfig(**data["config"])
        versions = [VersionArtifact(**item) for item in data["versions"]]
        return cls(
            snapshot_id=data["snapshot_id"],
            config=config,
            versions=versions,
            block_count=int(data["block_count"]),
            manifest_sha256=str(data.get("manifest_sha256") or ""),
            schema_version=int(data.get("schema_version") or 2),
        )


def write_manifest_atomic(manifest: SnapshotManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def read_manifest(path: Path) -> SnapshotManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = SnapshotManifest.from_dict(data)
    canonical = json.dumps(
        manifest.to_dict(exclude_sha=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if manifest.manifest_sha256 != expected:
        raise ValueError("manifest hash mismatch")
    return manifest
