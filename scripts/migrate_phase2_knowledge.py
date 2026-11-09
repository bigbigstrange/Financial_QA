#!/usr/bin/env python3
"""Side-by-side phase 2 candidate migration with an explicit activation gate."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.base import Base
from app.db.schema_upgrades import run_schema_upgrades
from app.services.document_catalog import DocumentCatalog
from app.services.document_service import DocumentService
from src.pipeline.file_validation import FileLimits, validate_input_file
from src.pipeline.identity import ParseContext, stable_digest
from src.pipeline.parsers import build_default_parsers
from src.pipeline.quality import assess_snapshot, process_document, write_json_atomic
from src.pipeline.snapshots import (
    SnapshotConfig,
    SnapshotManifest,
    VersionArtifact,
    sha256_file,
)


class MigrationGateError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class MigrationOptions:
    input_dir: str
    candidate_dir: str
    database_url: str
    activate: bool = False


@dataclass
class MigrationReport:
    candidate_snapshot_id: str
    document_count: int
    block_count: int
    duplicate_block_id_count: int
    missing_locator_count: int
    empty_block_count: int
    table_count: int
    table_row_group_count: int
    hard_failure_count: int
    failed_files: int
    status: str
    activation_requested: bool
    activated: bool = False

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def run_migration(options: MigrationOptions) -> MigrationReport:
    input_path = Path(options.input_dir)
    candidate_dir = Path(options.candidate_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    parsers = build_default_parsers()
    extensions = {
        ext
        for parser in parsers
        for ext in getattr(parser, "SUPPORTED_EXTS", set())
    }
    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )

    processed = []
    failures: list[dict] = []
    for file_path in files:
        relative = file_path.relative_to(input_path).as_posix()
        document_id = "mig_" + stable_digest(relative.lower())[:24]
        context = ParseContext.for_file(
            file_path,
            document_id,
            parser_config_hash="parser-v2",
            chunker_version="chunk-v2",
        )
        try:
            validate_input_file(file_path, FileLimits())
            item = process_document(
                file_path,
                candidate_dir / "docir",
                context,
                parsers,
            )
            processed.append(item)
        except Exception as exc:
            failures.append(
                {
                    "file": file_path.name,
                    "error": getattr(exc, "code", "parse_failed"),
                    "message": str(exc),
                }
            )

    artifacts = []
    for item in processed:
        artifacts.append(
            VersionArtifact(
                document_id=item.document.document_id,
                document_version=item.document.document_version,
                docir_path=str(item.docir_path),
                block_count=len(item.document.all_blocks()),
                docir_sha256=sha256_file(item.docir_path),
            )
        )
    config = SnapshotConfig(2, "chunk-v2", "parsers-v2")
    manifest = SnapshotManifest.build(artifacts, config)
    aggregate = assess_snapshot(
        [item.document for item in processed],
        snapshot_id=manifest.snapshot_id,
    )
    report = MigrationReport(
        candidate_snapshot_id=manifest.snapshot_id,
        document_count=len(processed),
        block_count=manifest.block_count,
        duplicate_block_id_count=aggregate.duplicate_block_id_count,
        missing_locator_count=aggregate.missing_locator_count,
        empty_block_count=aggregate.empty_block_count,
        table_count=aggregate.table_count,
        table_row_group_count=aggregate.table_row_group_count,
        hard_failure_count=len(failures) + aggregate.hard_failure_count,
        failed_files=len(failures),
        status="failed" if (len(failures) or aggregate.hard_failure_count) else "passed_with_warnings",
        activation_requested=options.activate,
    )
    report_dir = candidate_dir / manifest.snapshot_id
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        report_dir / "migration_report.json",
        report.to_dict(),
    )

    if options.activate:
        if report.hard_failure_count:
            raise MigrationGateError(
                "candidate_quality_failed",
                "候选快照质量门禁未通过，禁止激活",
            )
        _activate_candidate(
            options.database_url,
            candidate_dir,
            processed,
        )
        report.activated = True
        write_json_atomic(
            report_dir / "migration_report.json",
            report.to_dict(),
        )
    return report


def _activate_candidate(
    database_url: str,
    candidate_dir: Path,
    processed,
) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(database_url, future=True)
    try:
        Base.metadata.create_all(bind=engine)
        run_schema_upgrades(engine)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        db = SessionLocal()
        try:
            service = DocumentService(db, artifact_root=candidate_dir)
            for item in processed:
                service.ingest(
                    item.file_path,
                    document_id=item.document.document_id,
                    display_name=item.document.source_name or item.file_path.name,
                )
            db.commit()
            active = DocumentCatalog(db).active_snapshot()
            if active is None:
                raise MigrationGateError(
                    "activation_failed",
                    "激活后未产生 active 快照",
                )
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 2 候选迁移")
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    report = run_migration(
        MigrationOptions(
            input_dir=args.input,
            candidate_dir=args.candidate_dir,
            database_url=args.database_url,
            activate=args.activate,
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
