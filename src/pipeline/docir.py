from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocBlock:
    block_id: str
    block_type: str
    text: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    layout_type: Optional[str] = None
    table_md: Optional[str] = None
    table_html: Optional[str] = None
    formula_latex: Optional[str] = None
    image_ref: Optional[str] = None
    source_region_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    section_path: Optional[str] = None
    parent_id: Optional[str] = None
    caption: Optional[str] = None
    table_headers: List[str] = field(default_factory=list)
    table_rows: List[List[Any]] = field(default_factory=list)
    unit: Optional[str] = None
    ocr_used: bool = False
    quality_flags: List[str] = field(default_factory=list)
    element_index: Optional[int] = None
    structural_locator: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocBlock":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class DocPage:
    page_no: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    layout_regions: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[DocBlock] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocPage":
        known = {f.name for f in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known}
        payload["blocks"] = [DocBlock.from_dict(item) for item in data.get("blocks", [])]
        return cls(**payload)


def block_has_content(block: "DocBlock") -> bool:
    """块是否携带可检索内容。

    表格父块与图表块是**结构化载体**：即使 `text` 为空，只要它们带着
    `table_rows`（父表结构）或 `figure_ref`（图表裁剪图），就不是"空块"。
    旧知识库正是因为把 `text=""` 的表格块一律跳过，导致表格完全进不了索引。
    """
    if (block.text or "").strip() or (block.table_md or "").strip():
        return True
    if block.block_type == "table" and block.table_rows:
        return True
    if block.block_type in {"figure", "figure_caption"}:
        meta = block.meta or {}
        return bool(block.caption or meta.get("figure_ref") or meta.get("figure_text"))
    return False


@dataclass
class DocIR:
    doc_id: str
    source_path: str
    source_format: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pages: List[DocPage] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    document_version: str = "legacy-v1"
    source_hash: Optional[str] = None
    parser_name: Optional[str] = None
    parser_version: Optional[str] = None
    language: Optional[str] = None
    access_scope: List[str] = field(default_factory=list)
    schema_version: int = 2
    logical_document_id: Optional[str] = None
    source_name: Optional[str] = None
    source_size_bytes: int = 0
    parser_config_hash: Optional[str] = None
    chunker_version: Optional[str] = None

    @property
    def document_id(self) -> str:
        return self.logical_document_id or self.doc_id

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def add_page(self, page: DocPage) -> None:
        self.pages.append(page)

    def all_blocks(self) -> List[DocBlock]:
        return [block for page in self.pages for block in page.blocks]

    def require_block(self, block_id: str) -> DocBlock:
        for block in self.all_blocks():
            if block.block_id == block_id:
                return block
        raise KeyError(f"block not found: {block_id}")

    def validate_v2(self) -> List[str]:
        failures: List[str] = []
        if not self.document_id:
            failures.append("missing_document_id")
        if not self.document_version:
            failures.append("missing_document_version")
        blocks = self.all_blocks()
        if not blocks:
            failures.append("no_blocks")

        ids: Dict[str, int] = {}
        for block in blocks:
            ids[block.block_id] = ids.get(block.block_id, 0) + 1
            if not block.structural_locator:
                failures.append("missing_structural_locator")
            if not block_has_content(block):
                failures.append("empty_block")
            if block.block_type == "table_rows":
                if not block.parent_id:
                    failures.append("table_row_group_missing_parent")
                if not block.table_headers:
                    failures.append("table_row_group_missing_headers")
                if not block.table_rows:
                    failures.append("table_row_group_missing_data")
        duplicates = [bid for bid, count in ids.items() if count > 1]
        if duplicates:
            failures.append("duplicate_block_id")
        return sorted(set(failures))

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# {self.doc_id}")
        lines.append("")
        lines.append(f"- source: {self.source_path}")
        lines.append(f"- format: {self.source_format}")
        lines.append(f"- pages: {self.total_pages}")
        lines.append("")

        for page in self.pages:
            label = "Document" if page.page_no is None else f"Page {page.page_no}"
            lines.append(f"## {label}")
            lines.append("")
            for block in page.blocks:
                if block.text.strip():
                    lines.append(block.text.strip())
                    lines.append("")
                if block.table_md:
                    lines.append(block.table_md.strip())
                    lines.append("")
                if block.formula_latex:
                    lines.append(f"`{block.formula_latex.strip()}`")
                    lines.append("")
        return "\n".join(lines).strip() + "\n"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["document_id"] = self.document_id
        data["total_pages"] = self.total_pages
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocIR":
        known = {f.name for f in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known}
        payload["pages"] = [DocPage.from_dict(item) for item in data.get("pages", [])]
        return cls(**payload)

    def save(self, output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self.doc_id}.docir.json"
        md_path = output_dir / f"{self.doc_id}.md"

        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}
