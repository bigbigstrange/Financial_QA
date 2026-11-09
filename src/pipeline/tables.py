"""Deterministic table normalization and row-group chunking."""
from __future__ import annotations

from typing import Any, List, Optional

from .docir import DocBlock
from .identity import make_block_id


def serialize_group(
    section_path: Optional[str],
    caption: Optional[str],
    unit: Optional[str],
    headers: List[str],
    rows: List[List[str]],
    start_row: int,
) -> str:
    lines: List[str] = []
    if section_path:
        lines.append(f"[章节] {section_path}")
    if caption:
        lines.append(f"[表格] {caption}")
    if unit:
        lines.append(f"[单位] {unit}")
    lines.append("[列] " + " | ".join(headers))
    lines.extend(
        f"[行 {start_row + offset}] " + " | ".join(row)
        for offset, row in enumerate(rows)
    )
    return "\n".join(lines)


def normalize_table(
    rows: List[List[Any]],
    *,
    document_version: str,
    locator: str,
    section_path: Optional[str] = None,
    caption: Optional[str] = None,
    unit: Optional[str] = None,
    max_rows: int = 20,
    max_chars: int = 6000,
    page: Optional[int] = None,
) -> List[DocBlock]:
    normalized = [[_cell_text(cell) for cell in row] for row in rows]
    if not normalized:
        return []

    header_rows, data_rows = _split_headers(normalized)
    headers = _flatten_headers(header_rows)
    if not headers:
        headers = [f"列{i + 1}" for i in range(len(normalized[0]))]

    inferred = len(header_rows) == 1
    parent = DocBlock(
        block_id=make_block_id(document_version, locator, "table"),
        block_type="table",
        text=caption or "",
        page=page,
        section_path=section_path,
        structural_locator=locator,
        caption=caption,
        unit=unit,
        table_headers=headers,
        table_rows=normalized,
        meta={"header_rows": header_rows},
        quality_flags=["inferred_header"] if inferred else [],
    )
    blocks: List[DocBlock] = [parent]

    absolute_start = len(header_rows) + 1
    offset = 0
    for group in _split_rows(data_rows, max_rows, max_chars):
        group_start = absolute_start + offset
        group_end = group_start + len(group) - 1
        child_locator = f"{locator}/rows/{group_start:06d}-{group_end:06d}"
        child = DocBlock(
            block_id=make_block_id(document_version, child_locator, "table_rows"),
            block_type="table_rows",
            text=serialize_group(
                section_path,
                caption,
                unit,
                headers,
                group,
                group_start,
            ),
            page=page,
            parent_id=parent.block_id,
            section_path=section_path,
            structural_locator=child_locator,
            caption=caption,
            unit=unit,
            table_headers=headers,
            table_rows=group,
            meta={
                "header_rows": header_rows,
                "row_start": group_start,
                "row_end": group_end,
            },
            quality_flags=["inferred_header"] if inferred else [],
        )
        blocks.append(child)
        offset += len(group)
    return blocks


def _cell_text(cell: Any) -> str:
    text = str(cell).strip()
    return text.replace("\n", " ").replace("\r", " ")


def _split_headers(rows: List[List[str]]) -> tuple[List[List[str]], List[List[str]]]:
    if not rows:
        return [], []
    first = rows[0]
    if len(rows) >= 2 and any(not c for c in first) and any(rows[1]):
        return rows[:2], rows[2:]
    return [first], rows[1:]


def _flatten_headers(header_rows: List[List[str]]) -> List[str]:
    filled: List[List[str]] = []
    for row in header_rows:
        current: List[str] = []
        previous = ""
        for cell in row:
            cell = cell.strip()
            if not cell:
                cell = previous
            else:
                previous = cell
            current.append(cell)
        filled.append(current)

    flattened: List[str] = []
    width = max((len(row) for row in filled), default=0)
    for col in range(width):
        parts = [row[col] for row in filled if col < len(row) and row[col]]
        flattened.append("/".join(parts) if parts else f"列{col + 1}")
    return flattened


def _split_rows(
    rows: List[List[str]],
    max_rows: int,
    max_chars: int,
) -> List[List[List[str]]]:
    groups: List[List[List[str]]] = []
    current: List[List[str]] = []
    current_chars = 0
    for row in rows:
        line = " | ".join(row)
        if current and (len(current) >= max_rows or current_chars + len(line) > max_chars):
            groups.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += len(line) + 1
    if current:
        groups.append(current)
    return groups
