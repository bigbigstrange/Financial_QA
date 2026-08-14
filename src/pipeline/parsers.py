from abc import ABC, abstractmethod
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .docir import DocBlock, DocIR, DocPage


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

    def parse(self, file_path: Path) -> DocIR:
        content = self._read_text(file_path)
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            meta={"parser": "plain_text"},
        )
        doc.add_page(
            DocPage(
                page_no=1,
                blocks=[
                    DocBlock(
                        block_id=f"{file_path.stem}_p1_b1",
                        block_type="paragraph",
                        text=content,
                        page=1,
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

    def parse(self, file_path: Path) -> DocIR:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("缺少 python-docx 依赖") from exc

        document = Document(str(file_path))
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            meta={"parser": "word_structured"},
        )

        page = DocPage(page_no=1)
        block_idx = 1
        for p in document.paragraphs:
            text = (p.text or "").strip()
            if not text:
                continue
            style_name = (p.style.name or "").lower() if p.style else ""
            if style_name.startswith("heading"):
                block_type = "heading"
            elif style_name.startswith("list"):
                block_type = "list_item"
            else:
                block_type = "paragraph"
            formulas = _find_formula_candidates(text)
            page.blocks.append(
                DocBlock(
                    block_id=f"{file_path.stem}_p1_b{block_idx}",
                    block_type=block_type,
                    text=text,
                    page=1,
                    layout_type=block_type,
                    formula_latex=formulas[0] if formulas else None,
                    meta={
                        "style": p.style.name if p.style else "",
                        "formula_candidates": formulas,
                    },
                )
            )
            block_idx += 1

        for t_idx, table in enumerate(document.tables, start=1):
            table_rows = []
            for row in table.rows:
                table_rows.append([cell.text.strip() for cell in row.cells])
            table_md = _table_to_markdown(table_rows)
            page.blocks.append(
                DocBlock(
                    block_id=f"{file_path.stem}_p1_t{t_idx}",
                    block_type="table",
                    text="",
                    page=1,
                    layout_type="table",
                    table_md=table_md,
                    meta={"rows": table_rows},
                )
            )

        if not page.blocks:
            raise RuntimeError(f"Word parser produced empty result: {file_path}")
        doc.add_page(page)
        return doc


class PDFDocumentParser(BaseDocumentParser):
    SUPPORTED_EXTS = {".pdf"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: Path) -> DocIR:
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 PDF 解析依赖（PyMuPDF/pytesseract/Pillow）") from exc

        table_by_page = _extract_pdf_tables(str(file_path))
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            meta={"parser": "pdf_multimodal"},
        )

        pdf = fitz.open(file_path)
        try:
            for page_no, page in enumerate(pdf, start=1):
                blocks: List[DocBlock] = []
                text_blocks = page.get_text("blocks")
                for idx, blk in enumerate(text_blocks, start=1):
                    text = (blk[4] or "").strip()
                    if not text:
                        continue
                    block_type = "heading" if len(text) < 80 and text.count("\n") <= 1 else "paragraph"
                    formulas = _find_formula_candidates(text)
                    blocks.append(
                        DocBlock(
                            block_id=f"{file_path.stem}_p{page_no}_b{idx}",
                            block_type=block_type,
                            text=text,
                            page=page_no,
                            bbox=[float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3])],
                            layout_type=block_type,
                            formula_latex=formulas[0] if formulas else None,
                            source_region_id=f"pdf_text_region_{idx}",
                            meta={"formula_candidates": formulas},
                        )
                    )

                if not blocks:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = _safe_tesseract_ocr(pytesseract, img)
                    if ocr_text:
                        blocks.append(
                            DocBlock(
                                block_id=f"{file_path.stem}_p{page_no}_ocr",
                                block_type="paragraph",
                                text=ocr_text,
                                page=page_no,
                                layout_type="paragraph",
                                source_region_id="ocr_full_page",
                                meta={"ocr_fallback": True},
                            )
                        )

                for t_idx, table_md in enumerate(table_by_page.get(page_no, []), start=1):
                    blocks.append(
                        DocBlock(
                            block_id=f"{file_path.stem}_p{page_no}_t{t_idx}",
                            block_type="table",
                            text="",
                            page=page_no,
                            layout_type="table",
                            table_md=table_md,
                            source_region_id=f"pdf_table_region_{t_idx}",
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

    def parse(self, file_path: Path) -> DocIR:
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
        block = DocBlock(
            block_id=f"{file_path.stem}_p1_b1",
            block_type="paragraph",
            text=text,
            page=1,
            bbox=[0.0, 0.0, float(width), float(height)],
            layout_type="paragraph",
            formula_latex=formulas[0] if formulas else None,
            source_region_id="image_full_page",
            meta={"formula_candidates": formulas},
        )
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=file_path.suffix.lower(),
            meta={"parser": "image_ocr_layout"},
        )
        doc.add_page(
            DocPage(
                page_no=1,
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

    def parse(self, file_path: Path) -> DocIR:
        suffix = file_path.suffix.lower()
        sheets = self._read_xlsx(file_path) if suffix == ".xlsx" else self._read_xls(file_path)
        if not sheets:
            raise RuntimeError(f"Excel parser produced empty result: {file_path}")

        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=suffix,
            meta={"parser": "excel_sheets"},
        )
        page = DocPage(page_no=1)
        for idx, (sheet_name, rows) in enumerate(sheets, start=1):
            page.blocks.append(
                DocBlock(
                    block_id=f"{file_path.stem}_p1_h{idx}",
                    block_type="heading",
                    text=f"工作表: {sheet_name}",
                    page=1,
                    layout_type="heading",
                )
            )
            table_md = _table_to_markdown(rows)
            if table_md:
                page.blocks.append(
                    DocBlock(
                        block_id=f"{file_path.stem}_p1_t{idx}",
                        block_type="table",
                        text="",
                        page=1,
                        layout_type="table",
                        table_md=table_md,
                        meta={"sheet": sheet_name, "rows": [[str(c) for c in r] for r in rows[:50]]},
                    )
                )
            else:
                # 无表结构时退化为逐行文本
                flat = "\n".join(
                    "\t".join(c for c in row if c).strip() for row in rows if any(row)
                ).strip()
                if flat:
                    page.blocks.append(
                        DocBlock(
                            block_id=f"{file_path.stem}_p1_b{idx}",
                            block_type="paragraph",
                            text=flat,
                            page=1,
                            layout_type="paragraph",
                            meta={"sheet": sheet_name},
                        )
                    )
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
                    out.append((ws.title, rows))
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
                    out.append((sheet.name, rows))
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
                    out.append((str(ws.Name), rows))
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

    def parse(self, file_path: Path) -> DocIR:
        suffix = file_path.suffix.lower()
        if suffix == ".pptx":
            slides = self._read_pptx(file_path)
            parser_name = "pptx_structured"
        else:
            slides = self._read_ppt_com(file_path)
            parser_name = "ppt_com"

        if not slides or all(not t.strip() for _, t in slides):
            raise RuntimeError(f"PowerPoint parser produced empty result: {file_path}")

        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=suffix,
            meta={"parser": parser_name},
        )
        for page_no, text in slides:
            text = (text or "").strip()
            if not text:
                continue
            doc.add_page(
                DocPage(
                    page_no=page_no,
                    blocks=[
                        DocBlock(
                            block_id=f"{file_path.stem}_p{page_no}_b1",
                            block_type="paragraph",
                            text=text,
                            page=page_no,
                            layout_type="paragraph",
                        )
                    ],
                )
            )
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
            slides.append((idx, "\n".join(parts)))
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
                slides.append((i, "\n".join(p for p in parts if p)))
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

    def parse(self, file_path: Path) -> DocIR:
        text = self._extract_via_word_com(file_path).strip()
        if not text:
            raise RuntimeError(f"Word COM produced empty result: {file_path}")
        doc = DocIR(
            doc_id=file_path.stem,
            source_path=str(file_path),
            source_format=".doc",
            meta={"parser": "word_com_legacy"},
        )
        # 按空行粗切块，避免超长单块
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        if not chunks:
            chunks = [text]
        page = DocPage(page_no=1)
        for idx, chunk in enumerate(chunks[:200], start=1):
            page.blocks.append(
                DocBlock(
                    block_id=f"{file_path.stem}_p1_b{idx}",
                    block_type="paragraph",
                    text=chunk[:8000],
                    page=1,
                    layout_type="paragraph",
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

    def parse(self, file_path: Path) -> DocIR:
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
            meta={"parser": "rtf_striprtf"},
        )
        doc.add_page(
            DocPage(
                page_no=1,
                blocks=[
                    DocBlock(
                        block_id=f"{file_path.stem}_p1_b1",
                        block_type="paragraph",
                        text=text,
                        page=1,
                        layout_type="paragraph",
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
