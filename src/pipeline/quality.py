"""Parse quality reports and hard-gate validation for DocIR v2 artifacts."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .docir import DocIR, block_has_content
from .identity import ParseContext


@dataclass
class ParseQualityReport:
    document_id: str
    document_version: str
    parser_name: Optional[str]
    parser_version: Optional[str]
    page_or_element_count: int
    block_count: int
    block_type_counts: dict[str, int]
    table_count: int
    table_row_group_count: int
    ocr_page_count: int
    empty_block_count: int
    duplicate_block_id_count: int
    missing_locator_count: int
    missing_parent_count: int
    failed_page_or_element_count: int
    truncated_count: int
    uncertain_reading_order_count: int
    quality_flags: List[str]
    status: str
    failure_reasons: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SnapshotQualityReport:
    snapshot_id: Optional[str]
    document_count: int
    block_count: int
    duplicate_block_id_count: int
    missing_locator_count: int
    empty_block_count: int
    table_count: int
    table_row_group_count: int
    hard_failure_count: int
    status: str
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class QualityGateError(RuntimeError):
    def __init__(self, report: ParseQualityReport | SnapshotQualityReport) -> None:
        super().__init__(f"quality gate failed: {report.status}")
        self.report = report


@dataclass
class ProcessedDocument:
    file_path: Path
    document: DocIR
    report: ParseQualityReport
    docir_path: Path
    quality_path: Path


def assess_document(document: DocIR) -> ParseQualityReport:
    failures = list(document.validate_v2())
    blocks = document.all_blocks()
    ids: dict[str, int] = {}
    block_type_counts: dict[str, int] = {}
    empty_blocks = 0
    missing_locators = 0
    missing_parents = 0
    table_count = 0
    row_group_count = 0
    ocr_pages = 0
    truncated = 0
    uncertain_order = 0
    flags: List[str] = []

    for block in blocks:
        ids[block.block_id] = ids.get(block.block_id, 0) + 1
        block_type_counts[block.block_type] = (
            block_type_counts.get(block.block_type, 0) + 1
        )
        if not block.structural_locator:
            missing_locators += 1
        if not block_has_content(block):
            empty_blocks += 1
        if block.block_type == "table_rows":
            if not block.parent_id:
                missing_parents += 1
            if not block.table_headers or not block.table_rows:
                if "table_row_group_invalid" not in failures:
                    failures.append("table_row_group_invalid")
            row_group_count += 1
        if block.block_type == "table":
            table_count += 1
        if block.ocr_used:
            ocr_pages += 1
        flags.extend(block.quality_flags or [])
        if "uncertain_reading_order" in (block.quality_flags or []):
            uncertain_order += 1
        if "truncated_rows" in (block.quality_flags or []) or block.meta.get(
            "truncated"
        ):
            truncated += 1

    duplicates = sum(count - 1 for count in ids.values() if count > 1)
    if duplicates > 0 and "duplicate_block_id" not in failures:
        failures.append("duplicate_block_id")

    try:
        round_trip = DocIR.from_dict(json.loads(json.dumps(document.to_dict())))
        if round_trip.document_version != document.document_version:
            raise ValueError("version mismatch")
    except Exception:
        failures.append("serialization_roundtrip")

    warnings = int(uncertain_order > 0) + int(
        "inferred_header" in flags
    )
    status = "failed" if failures else (
        "passed_with_warnings" if warnings else "passed"
    )
    return ParseQualityReport(
        document_id=document.document_id,
        document_version=document.document_version,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        page_or_element_count=len(document.pages),
        block_count=len(blocks),
        block_type_counts=block_type_counts,
        table_count=table_count,
        table_row_group_count=row_group_count,
        ocr_page_count=ocr_pages,
        empty_block_count=empty_blocks,
        duplicate_block_id_count=duplicates,
        missing_locator_count=missing_locators,
        missing_parent_count=missing_parents,
        failed_page_or_element_count=0,
        truncated_count=truncated,
        uncertain_reading_order_count=uncertain_order,
        quality_flags=sorted(set(flags)),
        status=status,
        failure_reasons=failures,
    )


def assess_snapshot(
    documents: Iterable[DocIR],
    snapshot_id: Optional[str] = None,
) -> SnapshotQualityReport:
    reports = [assess_document(doc) for doc in documents]
    failures: List[str] = []
    duplicate = 0
    missing_locator = 0
    empty = 0
    tables = 0
    row_groups = 0
    block_count = 0
    for report in reports:
        block_count += report.block_count
        duplicate += report.duplicate_block_id_count
        missing_locator += report.missing_locator_count
        empty += report.empty_block_count
        tables += report.table_count
        row_groups += report.table_row_group_count
        if report.status == "failed":
            failures.extend(report.failure_reasons)
    hard_failures = len(failures)
    return SnapshotQualityReport(
        snapshot_id=snapshot_id,
        document_count=len(reports),
        block_count=block_count,
        duplicate_block_id_count=duplicate,
        missing_locator_count=missing_locator,
        empty_block_count=empty,
        table_count=tables,
        table_row_group_count=row_groups,
        hard_failure_count=hard_failures,
        status="failed" if hard_failures else "passed_with_warnings",
        failure_reasons=sorted(set(failures)),
    )


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def process_document(
    file_path: Path,
    output_dir: Path,
    context: ParseContext,
    parsers,
) -> ProcessedDocument:
    from .parsers import BaseDocumentParser

    candidates = [
        parser
        for parser in parsers
        if isinstance(parser, BaseDocumentParser) and parser.supports(file_path)
    ]
    if not candidates:
        raise QualityGateError(
            ParseQualityReport(
                document_id=context.document_id,
                document_version="no_parser",
                parser_name=None,
                parser_version=None,
                page_or_element_count=0,
                block_count=0,
                block_type_counts={},
                table_count=0,
                table_row_group_count=0,
                ocr_page_count=0,
                empty_block_count=0,
                duplicate_block_id_count=0,
                missing_locator_count=0,
                missing_parent_count=0,
                failed_page_or_element_count=0,
                truncated_count=0,
                uncertain_reading_order_count=0,
                quality_flags=[],
                status="failed",
                failure_reasons=["no_parser"],
            )
        )

    document: DocIR | None = None
    last_error: Exception | None = None
    for parser in candidates:
        try:
            document = parser.parse(file_path, context=context)
            break
        except Exception as exc:
            last_error = exc
    if document is None:
        raise QualityGateError(
            _synthetic_report(context, "all_parsers_failed", str(last_error))
        )

    report = assess_document(document)
    if report.status == "failed":
        raise QualityGateError(report)

    version_dir = output_dir / document.document_version
    docir_path = version_dir / "docir.json"
    quality_path = version_dir / "parse_quality.json"
    write_json_atomic(docir_path, document.to_dict())
    write_json_atomic(quality_path, report.to_dict())
    return ProcessedDocument(
        file_path=file_path,
        document=document,
        report=report,
        docir_path=docir_path,
        quality_path=quality_path,
    )


def _synthetic_report(
    context: ParseContext,
    reason: str,
    message: Optional[str],
) -> ParseQualityReport:
    return ParseQualityReport(
        document_id=context.document_id,
        document_version="failed",
        parser_name=None,
        parser_version=None,
        page_or_element_count=0,
        block_count=0,
        block_type_counts={},
        table_count=0,
        table_row_group_count=0,
        ocr_page_count=0,
        empty_block_count=0,
        duplicate_block_id_count=0,
        missing_locator_count=0,
        missing_parent_count=0,
        failed_page_or_element_count=0,
        truncated_count=0,
        uncertain_reading_order_count=0,
        quality_flags=[],
        status="failed",
        failure_reasons=[reason, message or ""],
    )
