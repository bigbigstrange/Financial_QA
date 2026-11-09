from abc import ABC, abstractmethod
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .docir import DocBlock, DocIR, DocPage
from .identity import ParseContext, make_block_id, make_document_version
from .tables import normalize_table

_EXCEL_MAX_ROWS = 500


class BaseDocumentParser(ABC):
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, file_path: Path) -> DocIR:
        raise NotImplementedError


class PlainTextParser(BaseDocumentParser):
    SUPPORTED_EXTS = {".txt", ".md"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        content = self._read_text(file_path)
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            parser_name="plain_text",
            parser_version="2",
            meta={"parser": "plain_text"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "plain_text",
                "2",
            )
        legacy = context is None
        locator = "body/000000/paragraph"
        doc.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock(
                        block_id=make_block_id(
                            doc.document_version,
                            locator,
                            "paragraph",
                        ),
                        block_type="paragraph",
                        text=content,
                        page=None,
                        structural_locator=locator,
                        element_index=0,
                        quality_flags=["legacy_identity"] if legacy else [],
                    )
                ],
            )
        )
        return doc

    @staticmethod
    def _read_text(file_path: Path) -> str:
        for enc in ("utf-8", "gbk"):
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding="utf-8", errors="ignore")


class WordDocumentParser(BaseDocumentParser):
    SUPPORTED_EXTS = {".docx"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("缺少 python-docx 依赖") from exc

        document = Document(str(file_path))
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            parser_name="word_structured",
            parser_version="2",
            meta={"parser": "word_structured"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "word_structured",
                "2",
            )

        page = DocPage(page_no=None)
        heading_stack: List[str] = []
        self._walk_body(
            document.element.body,
            page,
            document,
            heading_stack,
            doc.document_version,
            context is not None,
            locator_prefix="body",
        )

        if not page.blocks:
            raise RuntimeError(f"Word parser produced empty result: {file_path}")
        doc.add_page(page)
        return doc

    def _walk_body(
        self,
        parent,
        page: DocPage,
        document,
        heading_stack: List[str],
        document_version: str,
        v2_identity: bool,
        locator_prefix: str,
    ) -> None:
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        legacy = not v2_identity
        for element_index, child in enumerate(parent.iterchildren()):
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, document)
                text = (paragraph.text or "").strip()
                style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
                block_type = self._paragraph_type(paragraph, style_name)
                section_path = "/".join(heading_stack)
                locator = f"{locator_prefix}/{element_index:06d}/{block_type}"
                if not text:
                    self._update_heading_stack(paragraph, style_name, heading_stack)
                    continue
                styles = self._heading_level(paragraph, style_name)
                if styles is not None:
                    heading_stack = heading_stack[: styles - 1] + [text]
                    section_path = "/".join(heading_stack)
                formulas = _find_formula_candidates(text)
                block = DocBlock(
                    block_id=make_block_id(document_version, locator, block_type),
                    block_type=block_type,
                    text=text,
                    page=None,
                    layout_type=block_type,
                    formula_latex=formulas[0] if formulas else None,
                    section_path=section_path,
                    structural_locator=locator,
                    element_index=element_index,
                    meta={
                        "style": paragraph.style.name if paragraph.style else "",
                        "formula_candidates": formulas,
                    },
                    quality_flags=["legacy_identity"] if legacy else [],
                )
                page.blocks.append(block)
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                section_path = "/".join(heading_stack)
                locator = f"{locator_prefix}/{element_index:06d}/table"
                rows = [
                    [cell.text.strip() for cell in row.cells]
                    for row in table.rows
                ]
                blocks = normalize_table(
                    rows,
                    document_version=document_version,
                    locator=locator,
                    section_path=section_path,
                    page=None,
                )
                for block in blocks:
                    if legacy:
                        block.quality_flags = list(
                            dict.fromkeys(block.quality_flags + ["legacy_identity"])
                        )
                    block.meta["element_index"] = element_index
                    page.blocks.append(block)
                self._walk_nested_table_cells(
                    table,
                    page,
                    document,
                    heading_stack,
                    document_version,
                    v2_identity,
                    locator,
                )

    def _walk_nested_table_cells(
        self,
        table,
        page: DocPage,
        document,
        heading_stack: List[str],
        document_version: str,
        v2_identity: bool,
        table_locator: str,
    ) -> None:
        from docx.oxml.table import CT_Tbl
        from docx.table import Table

        legacy = not v2_identity
        for row_index, row in enumerate(table.rows, start=1):
            for cell_index, cell in enumerate(row.cells, start=1):
                prefix = f"{table_locator}/cell/{row_index:06d}/{cell_index:06d}"
                for local_index, child in enumerate(cell._tc.iterchildren()):
                    if isinstance(child, CT_Tbl):
                        nested = Table(child, document)
                        locator = f"{prefix}/{local_index:06d}/table"
                        rows = [
                            [cell.text.strip() for cell in row.cells]
                            for row in nested.rows
                        ]
                        nested_blocks = normalize_table(
                            rows,
                            document_version=document_version,
                            locator=locator,
                            section_path="/".join(heading_stack),
                            page=None,
                        )
                        if legacy:
                            for block in nested_blocks:
                                block.quality_flags = list(
                                    dict.fromkeys(
                                        block.quality_flags + ["legacy_identity"]
                                    )
                                )
                        page.blocks.extend(nested_blocks)

    @staticmethod
    def _paragraph_type(paragraph, style_name: str) -> str:
        if style_name.startswith("heading"):
            return "heading"
        if style_name.startswith("list"):
            return "list_item"
        return "paragraph"

    @staticmethod
    def _heading_level(paragraph, style_name: str) -> Optional[int]:
        name = style_name
        match = re.search(r"(?:heading|标题)\s*(\d+)", name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _update_heading_stack(
        paragraph,
        style_name: str,
        heading_stack: List[str],
    ) -> None:
        text = (paragraph.text or "").strip()
        level = WordDocumentParser._heading_level(paragraph, style_name)
        if level is not None and text:
            heading_stack[:] = heading_stack[: level - 1] + [text]


class PDFDocumentParser(BaseDocumentParser):
    SUPPORTED_EXTS = {".pdf"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        try:
            import fitz
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 PDF 解析依赖（PyMuPDF/Pillow）") from exc

        table_by_page = _extract_pdf_tables(str(file_path))
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            parser_name="pdf_multimodal",
            parser_version="2",
            meta={"parser": "pdf_multimodal"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "pdf_multimodal",
                "2",
            )
        legacy = context is None

        pdf = fitz.open(file_path)
        try:
            for page_no, page in enumerate(pdf, start=1):
                blocks: List[DocBlock] = []
                text_blocks = page.get_text("blocks")
                table_marks = table_by_page.get(page_no, [])
                for idx, blk in enumerate(text_blocks, start=1):
                    text = (blk[4] or "").strip()
                    if not text:
                        continue
                    bbox = [round(float(blk[0]), 1), round(float(blk[1]), 1),
                            round(float(blk[2]), 1), round(float(blk[3]), 1)]
                    locator = (
                        f"page/{page_no:06d}/text/{idx:06d}/bbox/"
                        + ",".join(f"{value:.1f}" for value in bbox)
                    )
                    if any(_text_is_inside_table(text, table_mark) for table_mark in table_marks):
                        continue
                    block_type = "heading" if len(text) < 80 and text.count("\n") <= 1 else "paragraph"
                    formulas = _find_formula_candidates(text)
                    blocks.append(
                        DocBlock(
                            block_id=make_block_id(
                                doc.document_version,
                                locator,
                                block_type,
                            ),
                            block_type=block_type,
                            text=text,
                            page=page_no,
                            bbox=bbox,
                            layout_type=block_type,
                            formula_latex=formulas[0] if formulas else None,
                            source_region_id=f"pdf_text_region_{idx}",
                            structural_locator=locator,
                            element_index=idx,
                            meta={"formula_candidates": formulas},
                            quality_flags=["legacy_identity"] if legacy else [],
                        )
                    )

                if not blocks and _needs_ocr(text_blocks, float(page.rect.width * page.rect.height)):
                    try:
                        import pytesseract
                    except ImportError as exc:
                        raise RuntimeError("缺少 OCR 依赖（pytesseract）") from exc
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = _safe_tesseract_ocr(pytesseract, img)
                    if ocr_text:
                        locator = f"page/{page_no:06d}/ocr"
                        blocks.append(
                            DocBlock(
                                block_id=make_block_id(
                                    doc.document_version,
                                    locator,
                                    "paragraph",
                                ),
                                block_type="paragraph",
                                text=ocr_text,
                                page=page_no,
                                layout_type="paragraph",
                                source_region_id="ocr_full_page",
                                structural_locator=locator,
                                ocr_used=True,
                                meta={
                                    "ocr_fallback": True,
                                    "ocr_language": "chi_sim+eng",
                                },
                                quality_flags=["ocr_used"],
                            )
                        )

                for t_idx, table_md in enumerate(table_marks, start=1):
                    locator = f"page/{page_no:06d}/table/{t_idx:06d}"
                    rows = _markdown_table_rows(table_md)
                    if rows:
                        table_blocks = normalize_table(
                            rows,
                            document_version=doc.document_version,
                            locator=locator,
                            page=page_no,
                        )
                        for block in table_blocks:
                            block.table_md = table_md
                            block.source_region_id = f"pdf_table_region_{t_idx}"
                            if legacy:
                                block.quality_flags = list(
                                    dict.fromkeys(
                                        block.quality_flags + ["legacy_identity"]
                                    )
                                )
                            blocks.append(block)
                    else:
                        blocks.append(
                            DocBlock(
                                block_id=make_block_id(
                                    doc.document_version,
                                    locator,
                                    "table",
                                ),
                                block_type="table",
                                text="",
                                page=page_no,
                                layout_type="table",
                                table_md=table_md,
                                source_region_id=f"pdf_table_region_{t_idx}",
                                structural_locator=locator,
                                quality_flags=["legacy_identity"] if legacy else [],
                            )
                        )

                if blocks:
                    doc.add_page(
                        DocPage(
                            page_no=page_no,
                            width=int(page.rect.width),
                            height=int(page.rect.height),
                            blocks=blocks,
                        )
                    )
        finally:
            pdf.close()

        if doc.total_pages == 0:
            raise RuntimeError(f"PDF parser produced empty result: {file_path}")
        return doc


class ImageDocumentParser(BaseDocumentParser):
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少图片解析依赖（pytesseract/Pillow）") from exc

        with Image.open(file_path) as img:
            rgb = img.convert("RGB")
            text = _safe_tesseract_ocr(pytesseract, rgb)
            width, height = rgb.size

        formulas = _find_formula_candidates(text)
        locator = "image/000000/ocr"
        block = DocBlock(
            block_id=make_block_id(
                "legacy-v1" if context is None else make_document_version(
                    context,
                    "image_ocr_layout",
                    "2",
                ),
                locator,
                "paragraph",
            ),
            block_type="paragraph",
            text=text,
            page=None,
            bbox=[0.0, 0.0, float(width), float(height)],
            layout_type="paragraph",
            formula_latex=formulas[0] if formulas else None,
            source_region_id="image_full_page",
            structural_locator=locator,
            ocr_used=True,
            meta={
                "formula_candidates": formulas,
                "ocr_language": "chi_sim+eng",
            },
            quality_flags=["ocr_used"],
        )
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            parser_name="image_ocr_layout",
            parser_version="2",
            meta={"parser": "image_ocr_layout"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "image_ocr_layout",
                "2",
            )
        if context is None:
            block.quality_flags = ["ocr_used", "legacy_identity"]
        doc.add_page(
            DocPage(
                page_no=None,
                width=width,
                height=height,
                layout_regions=[{"region_id": "image_full_page", "type": "paragraph"}],
                blocks=[block],
            )
        )
        return doc


def _safe_tesseract_ocr(pytesseract_module, image) -> str:
    """
    OCR 语言回退，避免因缺少 chi_sim 导致整页失败。
    """
    _configure_tesseract_cmd(pytesseract_module)
    last_err = None
    for lang in ("chi_sim+eng", "chi_sim", "eng"):
        try:
            out = pytesseract_module.image_to_string(image, lang=lang).strip()
            if out:
                return out
        except Exception as err:
            last_err = err
            continue
    if last_err:
        raise RuntimeError(f"Tesseract OCR 失败: {last_err}")
    return ""


def _configure_tesseract_cmd(pytesseract_module) -> None:
    """
    优先读取环境变量 TESSERACT_CMD，便于 Windows 显式指定 tesseract.exe。
    """
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract_module.pytesseract.tesseract_cmd = cmd

def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _sheet_segment(name: str) -> str:
    return str(name or "sheet").replace("/", "_").replace("\\", "_").strip()


def _text_is_inside_table(text: str, table_markdown: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized or len(normalized) < 2:
        return False
    table_text = _normalize_text(table_markdown)
    table_text = table_text.replace("|", " ").replace("-", " ")
    table_text = _normalize_text(table_text)
    return normalized in table_text


def _markdown_table_rows(markdown: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in (markdown or "").splitlines():
        if "|" not in line:
            continue
        stripped = line.strip()
        if stripped and set(stripped.replace("|", "").replace(" ", "")) == {"-"}:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _needs_ocr(
    text_blocks: List[tuple],
    page_area: float,
    min_chars: int = 20,
    min_coverage: float = 0.0005,
) -> bool:
    if not text_blocks:
        return True
    chars = sum(len((blk[4] or "").strip()) for blk in text_blocks)
    if chars < min_chars:
        return True
    coverage = 0.0
    for blk in text_blocks:
        width = max(0.0, float(blk[2]) - float(blk[0]))
        height = max(0.0, float(blk[3]) - float(blk[1]))
        coverage += width * height
    if page_area <= 0:
        return False
    return coverage / page_area < min_coverage


def _extract_pdf_tables(pdf_path: str) -> Dict[int, List[str]]:
    tables: Dict[int, List[str]] = {}
    try:
        import camelot
    except Exception:
        return tables

    try:
        extracted = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for tb in extracted:
            page_no = int(tb.page)
            rows = tb.df.fillna("").values.tolist()
            md = _table_to_markdown(rows)
            if md:
                tables.setdefault(page_no, []).append(md)
    except Exception:
        return tables
    return tables


def _table_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in rows]
    header = normalized[0]
    sep = ["---"] * max_cols
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _find_formula_candidates(text: str) -> List[str]:
    if not text:
        return []
    patterns = [
        r"[A-Za-z]+\s*=\s*[-+*/().A-Za-z0-9]+",
        r"\b\d+(?:\.\d+)?\s*[%‰]\b",
        r"\\frac\{.*?\}\{.*?\}",
    ]
    out = []
    for pat in patterns:
        out.extend(re.findall(pat, text))
    uniq = []
    for item in out:
        if item not in uniq:
            uniq.append(item)
    return uniq[:5]


class ExcelDocumentParser(BaseDocumentParser):
    """xlsx via openpyxl；xls via xlrd（必要时 Excel COM 回退）。"""

    SUPPORTED_EXTS = {".xlsx", ".xls"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        suffix = file_path.suffix.lower()
        sheets = self._read_xlsx(file_path) if suffix == ".xlsx" else self._read_xls(file_path)
        if not sheets:
            raise RuntimeError(f"Excel parser produced empty result: {file_path}")

        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=suffix,
            parser_name="excel_sheets",
            parser_version="2",
            meta={"parser": "excel_sheets"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "excel_sheets",
                "2",
            )
        legacy = context is None
        page = DocPage(page_no=None)
        for sheet_name, rows, truncated in sheets:
            segment = _sheet_segment(sheet_name)
            heading_locator = f"sheet/{segment}/heading"
            heading = DocBlock(
                block_id=make_block_id(
                    doc.document_version,
                    heading_locator,
                    "heading",
                ),
                block_type="heading",
                text=f"工作表: {sheet_name}",
                page=None,
                layout_type="heading",
                structural_locator=heading_locator,
                quality_flags=["legacy_identity"] if legacy else [],
            )
            if truncated:
                heading.quality_flags.append("truncated_rows")
            page.blocks.append(heading)
            table_md = _table_to_markdown(rows)
            if table_md:
                locator = f"sheet/{segment}/rows/000001-{len(rows):06d}"
                blocks = normalize_table(
                    rows,
                    document_version=doc.document_version,
                    locator=locator,
                    section_path=sheet_name,
                    page=None,
                )
                for block in blocks:
                    block.table_md = table_md
                    block.meta["sheet"] = sheet_name
                    if truncated:
                        block.quality_flags = list(
                            dict.fromkeys(block.quality_flags + ["truncated_rows"])
                        )
                    if legacy:
                        block.quality_flags = list(
                            dict.fromkeys(block.quality_flags + ["legacy_identity"])
                        )
                    page.blocks.append(block)
            else:
                flat = "\n".join(
                    "\t".join(c for c in row if c).strip() for row in rows if any(row)
                ).strip()
                if flat:
                    locator = f"sheet/{segment}/text/000001"
                    page.blocks.append(
                        DocBlock(
                            block_id=make_block_id(
                                doc.document_version,
                                locator,
                                "paragraph",
                            ),
                            block_type="paragraph",
                            text=flat,
                            page=None,
                            layout_type="paragraph",
                            meta={"sheet": sheet_name},
                            structural_locator=locator,
                            quality_flags=["legacy_identity"] if legacy else [],
                        )
                    )
                    if truncated:
                        page.blocks[-1].quality_flags.append("truncated_rows")
        if not page.blocks:
            raise RuntimeError(f"Excel parser produced empty result: {file_path}")
        doc.add_page(page)
        return doc

    @staticmethod
    def _normalize_rows(raw_rows: List[List[object]], max_rows: int = 500) -> List[List[str]]:
        rows: List[List[str]] = []
        for row in raw_rows[:max_rows]:
            cells = [("" if c is None else str(c).strip()) for c in row]
            if any(cells):
                rows.append(cells)
        return rows

    def _read_xlsx(self, file_path: Path) -> List[tuple]:
        from openpyxl import load_workbook

        wb = load_workbook(str(file_path), read_only=True, data_only=True)
        out = []
        try:
            for ws in wb.worksheets:
                raw = []
                for row in ws.iter_rows(values_only=True):
                    raw.append(list(row))
                rows = self._normalize_rows(raw)
                if rows:
                    out.append((ws.title, rows, len(raw) > _EXCEL_MAX_ROWS))
        finally:
            wb.close()
        return out

    def _read_xls(self, file_path: Path) -> List[tuple]:
        try:
            import xlrd

            book = xlrd.open_workbook(str(file_path))
            out = []
            for sheet in book.sheets():
                raw = [sheet.row_values(r) for r in range(sheet.nrows)]
                rows = self._normalize_rows(raw)
                if rows:
                    out.append((sheet.name, rows, sheet.nrows > _EXCEL_MAX_ROWS))
            if out:
                return out
        except Exception:
            pass
        return self._read_xls_via_excel_com(file_path)

    def _read_xls_via_excel_com(self, file_path: Path) -> List[tuple]:
        excel = None
        wb = None
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            excel = win32com.client.DispatchEx("Excel.Application")
            excel.Visible = False
            excel.DisplayAlerts = False
            wb = excel.Workbooks.Open(str(file_path.resolve()), ReadOnly=True)
            out = []
            for i in range(1, wb.Worksheets.Count + 1):
                ws = wb.Worksheets(i)
                used = ws.UsedRange
                if used is None:
                    continue
                values = used.Value
                raw: List[List[object]] = []
                if values is None:
                    continue
                if not isinstance(values, tuple):
                    raw = [[values]]
                else:
                    for row in values:
                        if isinstance(row, tuple):
                            raw.append(list(row))
                        else:
                            raw.append([row])
                rows = self._normalize_rows(raw)
                if rows:
                    out.append((str(ws.Name), rows, len(raw) > _EXCEL_MAX_ROWS))
            return out
        except Exception as exc:
            raise RuntimeError(f"Excel COM 解析失败: {file_path}: {exc}") from exc
        finally:
            try:
                if wb is not None:
                    wb.Close(False)
            except Exception:
                pass
            try:
                if excel is not None:
                    excel.Quit()
            except Exception:
                pass
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass


class PowerPointDocumentParser(BaseDocumentParser):
    """pptx 用 python-pptx；旧版 ppt/pps 走 PowerPoint COM。"""

    SUPPORTED_EXTS = {".pptx", ".ppt", ".pps"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        suffix = file_path.suffix.lower()
        if suffix == ".pptx":
            slides = self._read_pptx(file_path)
            parser_name = "pptx_structured"
        else:
            slides = self._read_ppt_com(file_path)
            parser_name = "ppt_com"

        if not slides or all(not parts for _, parts in slides):
            raise RuntimeError(f"PowerPoint parser produced empty result: {file_path}")

        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=suffix,
            parser_name=parser_name,
            parser_version="2",
            meta={"parser": parser_name},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                parser_name,
                "2",
            )
        legacy = context is None
        for page_no, parts in slides:
            if not parts:
                continue
            page = DocPage(page_no=page_no)
            for shape_index, part in enumerate(parts, start=1):
                part = (part or "").strip()
                if not part:
                    continue
                if part.startswith("|"):
                    rows = _markdown_table_rows(part)
                    locator = f"slide/{page_no:06d}/shape/{shape_index:06d}/table"
                    blocks = normalize_table(
                        rows,
                        document_version=doc.document_version,
                        locator=locator,
                        page=page_no,
                    )
                    for block in blocks:
                        block.table_md = part
                        if legacy:
                            block.quality_flags = list(
                                dict.fromkeys(
                                    block.quality_flags + ["legacy_identity"]
                                )
                            )
                        page.blocks.append(block)
                else:
                    locator = f"slide/{page_no:06d}/shape/{shape_index:06d}"
                    page.blocks.append(
                        DocBlock(
                            block_id=make_block_id(
                                doc.document_version,
                                locator,
                                "paragraph",
                            ),
                            block_type="paragraph",
                            text=part,
                            page=page_no,
                            layout_type="paragraph",
                            structural_locator=locator,
                            element_index=shape_index,
                            quality_flags=["legacy_identity"] if legacy else [],
                        )
                    )
            if page.blocks:
                doc.add_page(page)
        if doc.total_pages == 0:
            raise RuntimeError(f"PowerPoint parser produced empty result: {file_path}")
        return doc

    @staticmethod
    def _read_pptx(file_path: Path) -> List[tuple]:
        from pptx import Presentation

        prs = Presentation(str(file_path))
        slides = []
        for idx, slide in enumerate(prs.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    for para in shape.text_frame.paragraphs:
                        line = "".join(run.text for run in para.runs).strip()
                        if line:
                            parts.append(line)
                if getattr(shape, "has_table", False):
                    table_rows = []
                    for row in shape.table.rows:
                        table_rows.append([cell.text.strip() for cell in row.cells])
                    md = _table_to_markdown(table_rows)
                    if md:
                        parts.append(md)
            slides.append((idx, parts))
        return slides

    @staticmethod
    def _read_ppt_com(file_path: Path) -> List[tuple]:
        ppt_app = None
        prs = None
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            ppt_app = win32com.client.DispatchEx("PowerPoint.Application")
            # 部分环境 WithWindow=False 更稳
            prs = ppt_app.Presentations.Open(str(file_path.resolve()), WithWindow=False)
            slides = []
            for i in range(1, prs.Slides.Count + 1):
                slide = prs.Slides(i)
                parts = []
                for j in range(1, slide.Shapes.Count + 1):
                    shape = slide.Shapes(j)
                    try:
                        if shape.HasTextFrame:
                            t = shape.TextFrame.TextRange.Text
                            if t:
                                parts.append(str(t).replace("\r", "\n").strip())
                    except Exception:
                        continue
                slides.append((i, [p for p in parts if p]))
            return slides
        except Exception as exc:
            raise RuntimeError(f"PowerPoint COM 解析失败: {file_path}: {exc}") from exc
        finally:
            try:
                if prs is not None:
                    prs.Close()
            except Exception:
                pass
            try:
                if ppt_app is not None:
                    ppt_app.Quit()
            except Exception:
                pass
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass


class LegacyWordDocParser(BaseDocumentParser):
    """旧版 .doc：通过 Word COM 抽正文（本机已装 Office）。"""

    SUPPORTED_EXTS = {".doc"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        text = self._extract_via_word_com(file_path).strip()
        if not text:
            raise RuntimeError(f"Word COM produced empty result: {file_path}")
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=".doc",
            parser_name="word_com_legacy",
            parser_version="2",
            meta={"parser": "word_com_legacy"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "word_com_legacy",
                "2",
            )
        legacy = context is None
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        if not chunks:
            chunks = [text]
        page = DocPage(page_no=None)
        for idx, chunk in enumerate(chunks[:200], start=1):
            locator = f"body/{idx:06d}/paragraph"
            page.blocks.append(
                DocBlock(
                    block_id=make_block_id(
                        doc.document_version,
                        locator,
                        "paragraph",
                    ),
                    block_type="paragraph",
                    text=chunk[:8000],
                    page=None,
                    layout_type="paragraph",
                    structural_locator=locator,
                    element_index=idx,
                    quality_flags=["legacy_layout"]
                    + (["legacy_identity"] if legacy else []),
                )
            )
        doc.add_page(page)
        return doc

    @staticmethod
    def _extract_via_word_com(file_path: Path) -> str:
        word = None
        document = None
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            document = word.Documents.Open(str(file_path.resolve()), ReadOnly=True)
            return str(document.Content.Text or "")
        except Exception as exc:
            raise RuntimeError(f"Word COM 解析失败: {file_path}: {exc}") from exc
        finally:
            try:
                if document is not None:
                    document.Close(False)
            except Exception:
                pass
            try:
                if word is not None:
                    word.Quit()
            except Exception:
                pass
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass


class RtfDocumentParser(BaseDocumentParser):
    SUPPORTED_EXTS = {".rtf"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path, *, context: Optional[ParseContext] = None) -> DocIR:
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as exc:
            raise RuntimeError("缺少 striprtf 依赖") from exc

        raw = None
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                raw = file_path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        text = (rtf_to_text(raw) or "").strip()
        if not text:
            raise RuntimeError(f"RTF parser produced empty result: {file_path}")

        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=".rtf",
            parser_name="rtf_striprtf",
            parser_version="2",
            meta={"parser": "rtf_striprtf"},
        )
        if context is not None:
            doc.logical_document_id = context.document_id
            doc.source_name = context.source_name
            doc.source_hash = context.source_sha256
            doc.source_size_bytes = context.source_size_bytes
            doc.parser_config_hash = context.parser_config_hash
            doc.chunker_version = context.chunker_version
            doc.document_version = make_document_version(
                context,
                "rtf_striprtf",
                "2",
            )
        legacy = context is None
        locator = "body/000000/paragraph"
        doc.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock(
                        block_id=make_block_id(
                            doc.document_version,
                            locator,
                            "paragraph",
                        ),
                        block_type="paragraph",
                        text=text,
                        page=None,
                        layout_type="paragraph",
                        structural_locator=locator,
                        element_index=0,
                        quality_flags=["legacy_layout"]
                        + (["legacy_identity"] if legacy else []),
                    )
                ],
            )
        )
        return doc


def build_default_parsers(legacy_output_dir: Optional[Path] = None) -> List[BaseDocumentParser]:
    _ = legacy_output_dir
    return [
        PlainTextParser(),
        WordDocumentParser(),
        LegacyWordDocParser(),
        PDFDocumentParser(),
        ImageDocumentParser(),
        ExcelDocumentParser(),
        PowerPointDocumentParser(),
        RtfDocumentParser(),
    ]
