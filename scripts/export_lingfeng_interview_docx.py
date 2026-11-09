# -*- coding: utf-8 -*-
"""Export 凌风云链 interview script markdown to Word (.docx)."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "凌风云链-AI全栈工程师实习-张艳面试稿.md"
OUT_PATH = ROOT / "凌风云链-AI全栈工程师实习-张艳面试稿.docx"


def set_run_font(run, name: str = "微软雅黑", size: int = 11, bold: bool = False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color


def add_formatted_paragraph(doc: Document, text: str, style: str = "Normal", indent: bool = False):
    p = doc.add_paragraph(style=style)
    if indent:
        p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(6)

    pattern = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            run = p.add_run(text[pos : m.start()])
            set_run_font(run)
        if m.group(1):
            run = p.add_run(m.group(1))
            set_run_font(run, bold=True)
        else:
            run = p.add_run(m.group(2))
            set_run_font(run, name="Consolas", size=10)
        pos = m.end()
    if pos < len(text):
        run = p.add_run(text[pos:])
        set_run_font(run)
    return p


def add_code_block(doc: Document, lines: list[str]):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    shading = p._element.get_or_add_pPr()
    for line in lines:
        if line.strip() == "" and len(lines) > 1:
            p.add_run("\n")
            continue
        run = p.add_run(line + ("\n" if line != lines[-1] else ""))
        set_run_font(run, name="Consolas", size=9, color=RGBColor(0x33, 0x33, 0x33))


def parse_table_block(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    if len(lines) < 2:
        return None
    if not all("|" in ln for ln in lines[:2]):
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    if not re.match(r"^[\|\s\-:]+$", lines[1]):
        return None
    rows = []
    for ln in lines[2:]:
        if "|" not in ln:
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return header, rows


def add_table(doc: Document, header: list[str], rows: list[list[str]]):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(header):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for run in p.runs:
                set_run_font(run, bold=True, size=10)
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = val
            for p in cells[c_idx].paragraphs:
                for run in p.runs:
                    set_run_font(run, size=10)
    doc.add_paragraph()


def convert_md_to_docx(md_text: str, out_path: Path):
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "微软雅黑"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    normal.font.size = Pt(11)

    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                add_code_block(doc, code_buf)
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped.startswith("# "):
            p = doc.add_heading(stripped[2:].strip(), level=0)
            for run in p.runs:
                set_run_font(run, size=18, bold=True)
            i += 1
            continue

        if stripped.startswith("## "):
            p = doc.add_heading(stripped[3:].strip(), level=1)
            for run in p.runs:
                set_run_font(run, size=15, bold=True)
            i += 1
            continue

        if stripped.startswith("### "):
            p = doc.add_heading(stripped[4:].strip(), level=2)
            for run in p.runs:
                set_run_font(run, size=13, bold=True)
            i += 1
            continue

        if stripped.startswith("#### "):
            p = doc.add_heading(stripped[5:].strip(), level=3)
            for run in p.runs:
                set_run_font(run, size=12, bold=True)
            i += 1
            continue

        if stripped.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            parsed = parse_table_block(tbl_lines)
            if parsed:
                add_table(doc, parsed[0], parsed[1])
            continue

        if stripped.startswith(">"):
            quote = stripped.lstrip(">").strip()
            add_formatted_paragraph(doc, quote, indent=True)
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            bullet_text = re.sub(r"^[-*]\s+", "", stripped)
            bullet_text = re.sub(r"^\d+\.\s+", "", bullet_text)
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Pt(18)
            p.paragraph_format.space_after = Pt(3)
            pattern = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")
            pos = 0
            for m in pattern.finditer(bullet_text):
                if m.start() > pos:
                    run = p.add_run(bullet_text[pos : m.start()])
                    set_run_font(run)
                if m.group(1):
                    run = p.add_run(m.group(1))
                    set_run_font(run, bold=True)
                else:
                    run = p.add_run(m.group(2))
                    set_run_font(run, name="Consolas", size=10)
                pos = m.end()
            if pos < len(bullet_text):
                run = p.add_run(bullet_text[pos:])
                set_run_font(run)
            i += 1
            continue

        if stripped == "---":
            doc.add_paragraph()
            i += 1
            continue

        add_formatted_paragraph(doc, stripped)
        i += 1

    doc.save(str(out_path))
    print(f"Generated: {out_path}")


def main():
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Markdown source not found: {MD_PATH}")
    md_text = MD_PATH.read_text(encoding="utf-8")
    convert_md_to_docx(md_text, OUT_PATH)


if __name__ == "__main__":
    main()
