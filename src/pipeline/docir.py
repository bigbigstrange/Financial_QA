from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocBlock:
    block_id: str
    block_type: str
    text: str
    page: int
    bbox: Optional[List[float]] = None
    # 新增多模态结构化字段（阶段1）
    layout_type: Optional[str] = None
    table_md: Optional[str] = None
    table_html: Optional[str] = None
    formula_latex: Optional[str] = None
    image_ref: Optional[str] = None
    source_region_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocPage:
    page_no: int
    width: Optional[int] = None
    height: Optional[int] = None
    layout_regions: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[DocBlock] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocIR:
    doc_id: str
    source_path: str
    source_format: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pages: List[DocPage] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def add_page(self, page: DocPage) -> None:
        self.pages.append(page)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# {self.doc_id}")
        lines.append("")
        lines.append(f"- source: {self.source_path}")
        lines.append(f"- format: {self.source_format}")
        lines.append(f"- pages: {self.total_pages}")
        lines.append("")

        for page in self.pages:
            lines.append(f"## Page {page.page_no}")
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
        data["total_pages"] = self.total_pages
        return data

    def save(self, output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self.doc_id}.docir.json"
        md_path = output_dir / f"{self.doc_id}.md"

        json_path.write_text(
            __import__("json").dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}
