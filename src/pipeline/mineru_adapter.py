"""MinerU 4.x MiddleJson → DocIR v3 适配器。

MinerU 的产物契约（实测 4.0.5，schema `docvortex.middle` v2.0）：

    {
      "metadata":  {"file_suffix", "producer": {...}, "document": {...}},
      "extensions": {
        "mineru": {"tier", "parse_mode"},            # parse_mode ∈ txt | ocr
        "docvortex_layout": {"version": 1,
                             "pages": [{"page_idx", "width_pt", "height_pt"}]}
      },
      "pages": [{"page_idx": 0-based, "blocks": [...]}],
      "schema": "docvortex.middle", "schema_version": "2.0"
    }

    block = {
      "type": "doc_title|paragraph_title|text|table|chart|header|footer|page_number|index",
      "index": int,
      "bbox": [x0, y0, x1, y1],        # **归一化到 0-1**，需换算成 pt
      "level": int,                     # 仅 doc_title / paragraph_title
      "content": [segment, ...]         # 见下
    }

    segment:
      text  : {"type": "text", "content": "..."}
      table : {"type": "table_body", "image_path": "images/page_0_table_13.jpg",
               "content": "<table><tbody><tr><td>...</td></tr></tbody></table>"}
      chart : {"type": "chart_body",  "image_path": "images/page_0_chart_23.jpg",
               "content": ""}
              {"type": "chart_footnote", "content": [{"type": "text", "content": "资料来源:..."}]}

本适配器把 MinerU 的语义块映射为 DocIR v3 的 `DocBlock`，解决旧链路的四类缺陷：
  1. 块粒度过细（旧链路 83.5% 的块 < 50 字）→ 直接采用 MinerU 的段落级语义块；
  2. 表格完全没抽到（旧链路全库仅 236 个表）→ 解析 `table_body` 的 HTML 表格，
     复用 `src/pipeline/tables.py :: normalize_table()` 做表头继承与行组切块；
  3. 图表完全没处理（旧链路页图路径全部失效）→ 记录 `figure_ref`/`figure_kind`/`figure_caption`，
     并可用 PyMuPDF 渲染整页图，让视觉通道真正可用；
  4. 页眉/页脚/页码污染索引（旧链路 17% 无中文块）→ 显式识别为噪声，不进入检索单元。
"""
from __future__ import annotations

import html as _html
import json
import re
import shutil
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .docir import DocBlock, DocIR, DocPage
from .identity import ParseContext, make_block_id, make_document_version
from .tables import normalize_table

MINERU_PARSER_NAME = "mineru"
MINERU_PARSER_VERSION = "4.0.5"
MINERU_TIER_DEFAULT = "standard"

# 页眉/页脚/页码：不进检索单元，只计数（旧链路正是被这三类噪声污染）
NOISE_TYPES = {"header", "footer", "page_number"}

# MinerU 块类型 → DocIR block_type
TEXT_TYPES = {"text", "index"}
HEADING_TYPES = {"doc_title", "paragraph_title"}
TABLE_TYPES = {"table"}
FIGURE_TYPES = {"chart", "image"}


# MinerU 把表/图/表的附属文本也放在 content 数组里（不是独立 block）。
# 旧实现只取 *_body 段，把 caption/footnote 全丢了 —— 表格的标题与**单位**
# 正是金融问答必需的上下文，必须还原并交给 normalize_table。
TABLE_BODY_TYPES = {"table_body"}
TABLE_CAPTION_TYPES = {"table_caption"}
TABLE_FOOTNOTE_TYPES = {"table_footnote"}
FIGURE_BODY_TYPES = {"chart_body", "image_body"}
FIGURE_CAPTION_TYPES = {"chart_caption", "image_caption"}
FIGURE_FOOTNOTE_TYPES = {"chart_footnote", "image_footnote"}

# "单位:百万元人民币（百分比除外）" / "单位：万元" / "单位:%"
_UNIT_PATTERN = re.compile(r"^\s*单位\s*[:：]\s*(?P<unit>.+?)\s*$")


class MineruUnavailable(RuntimeError):
    """MinerU 不可用（未安装 / 调用失败 / 产物缺失）。调用方应回退到 PDFDocumentParser。"""


# ---------------------------------------------------------------------------
# bbox / HTML 工具
# ---------------------------------------------------------------------------

def _norm_to_pt(bbox: Sequence[Any], width_pt: float, height_pt: float) -> Optional[List[float]]:
    """归一化 bbox(0-1) → 页面坐标(pt)。非法输入返回 None。"""
    if not bbox or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return [
        round(x0 * width_pt, 1),
        round(y0 * height_pt, 1),
        round(x1 * width_pt, 1),
        round(y1 * height_pt, 1),
    ]


class _TableHTMLParser(HTMLParser):
    """把 MinerU 的 `<table>` HTML 解析成行列表，并保留合并单元格跨度。

    colspan 会以空串占位补足列数（保证每行列数一致）；
    rowspan 记录在 spans 里，不做纵向填充（避免伪造数据）。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self.spans: List[List[Tuple[int, int]]] = []
        self._row: Optional[List[str]] = None
        self._row_spans: List[Tuple[int, int]] = []
        self._cell: Optional[List[str]] = None
        self._colspan = 1
        self._rowspan = 1

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._row = []
            self._row_spans = []
        elif tag in ("td", "th"):
            self._cell = []
            attr = {k.lower(): (v or "") for k, v in attrs}
            try:
                self._colspan = max(1, int(attr.get("colspan") or 1))
            except ValueError:
                self._colspan = 1
            try:
                self._rowspan = max(1, int(attr.get("rowspan") or 1))
            except ValueError:
                self._rowspan = 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            if any(c.strip() for c in self._row):
                self.rows.append(self._row)
                self.spans.append(self._row_spans)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(text)
            self._row_spans.append((self._colspan, self._rowspan))
            for _ in range(self._colspan - 1):
                self._row.append("")
                self._row_spans.append((0, 0))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_rows(html: str) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    if not html or "<t" not in html.lower():
        return [], []
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — MinerU 偶发不闭合标签，尽力而为
        pass
    return parser.rows, parser.spans


def rows_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized = [list(r) + [""] * (width - len(r)) for r in rows]
    head = normalized[0]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _segment_text(segments: Any) -> str:
    """把 content 段里的纯文本拼起来（跳过 table_body / chart_body 这类非文本段）。"""
    parts: List[str] = []
    if isinstance(segments, str):
        return segments.strip()
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        if seg.get("type") in {"table_body", "chart_body"}:
            continue
        inner = seg.get("content")
        if isinstance(inner, str):
            parts.append(inner)
        elif isinstance(inner, list):
            parts.append(_segment_text(inner))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(p for p in parts if p)).strip()


def _find_segment(segments: Any, seg_type: str) -> Optional[Dict[str, Any]]:
    for seg in segments or []:
        if isinstance(seg, dict) and seg.get("type") == seg_type:
            return seg
    return None


def _segments_of(segments: Any, types: set[str]) -> List[Dict[str, Any]]:
    return [
        seg for seg in (segments or [])
        if isinstance(seg, dict) and seg.get("type") in types
    ]


def _segment_content_text(seg: Dict[str, Any]) -> str:
    """取段落文本；caption/footnote 段的 content 可能是嵌套的 text 段列表。"""
    inner = seg.get("content")
    if isinstance(inner, str):
        return inner.strip()
    if isinstance(inner, list):
        return _segment_text(inner)
    return ""


def split_table_segments(
    segments: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """从表格块的 content 数组里分离出 (body, caption, unit)。

    MinerU 的表格块形如：
        [{"type": "table_body",     "content": "<table>…", "image_path": …},
         {"type": "table_caption",  "content": [{"type": "text", "content": "金融资产投资情况"}]},
         {"type": "table_footnote", "content": [{"type": "text", "content": "单位:百万元人民币"}]},
         {"type": "table_footnote", "content": [{"type": "text", "content": "注: 1、…"}]}]

    返回的 unit 从 footnote 里按 `单位:xxx` 抽取；其余 footnote 与 caption
    合并成 caption 文本，保证表注不丢失。
    """
    body = _find_segment(segments, "table_body")
    if body is None:
        return None, None, None

    caption_parts: List[str] = []
    unit: Optional[str] = None
    for seg in _segments_of(segments, TABLE_CAPTION_TYPES):
        text = _segment_content_text(seg)
        if text:
            caption_parts.append(text)
    for seg in _segments_of(segments, TABLE_FOOTNOTE_TYPES):
        text = _segment_content_text(seg)
        if not text:
            continue
        match = _UNIT_PATTERN.match(text)
        if match and unit is None:
            unit = match.group("unit")
            continue
        caption_parts.append(text)
    caption = "\n".join(caption_parts).strip() or None
    return body, caption, unit


def split_figure_segments(
    segments: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """从图表块的 content 数组里分离出 (body, caption, footnote)。

    caption 优先取 `chart_caption`/`image_caption`，缺失时退回 footnote
    （很多研报只有 "资料来源:Wind" 这类脚注）。
    """
    body = next(
        (seg for seg in _segments_of(segments, FIGURE_BODY_TYPES)), None
    )
    if body is None:
        return None, None, None
    captions = [_segment_content_text(s) for s in _segments_of(segments, FIGURE_CAPTION_TYPES)]
    footnotes = [_segment_content_text(s) for s in _segments_of(segments, FIGURE_FOOTNOTE_TYPES)]
    caption = "\n".join(t for t in captions if t).strip() or None
    note = "\n".join(t for t in footnotes if t).strip() or None
    return body, caption, note


# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------

class _HeadingStack:
    """按 level 维护章节路径，产出 `A/B/C` 形式的 section_path。"""

    def __init__(self) -> None:
        self.stack: List[str] = []

    def push(self, level: int, text: str) -> str:
        level = max(1, min(int(level or 1), 6))
        self.stack = self.stack[: level - 1] + [text]
        return "/".join(self.stack)

    @property
    def path(self) -> str:
        return "/".join(self.stack)


def middle_json_to_docir(
    middle: Dict[str, Any],
    *,
    file_path: Path,
    parsed_dir: Path,
    context: ParseContext,
    tier: str = MINERU_TIER_DEFAULT,
    render_page_images: bool = True,
    page_image_dpi: int = 150,
) -> DocIR:
    """MinerU MiddleJson → DocIR v3。"""
    if not isinstance(middle, dict) or "pages" not in middle:
        raise MineruUnavailable("MiddleJson 结构不合法（缺少 pages）")

    doc = DocIR(
        doc_id=file_path.stem,
        source_path=str(file_path),
        source_format=file_path.suffix.lower(),
        parser_name=MINERU_PARSER_NAME,
        parser_version=MINERU_PARSER_VERSION,
        meta={
            "parser": MINERU_PARSER_NAME,
            "mineru_tier": tier,
            "mineru_parse_mode": ((middle.get("extensions") or {}).get("mineru") or {}).get(
                "parse_mode"
            ),
            "mineru_schema": middle.get("schema"),
            "mineru_schema_version": middle.get("schema_version"),
            "mineru_document": (middle.get("metadata") or {}).get("document"),
        },
    )
    doc.logical_document_id = context.document_id
    doc.source_name = context.source_name
    doc.source_hash = context.source_sha256
    doc.source_size_bytes = context.source_size_bytes
    doc.parser_config_hash = context.parser_config_hash
    doc.chunker_version = context.chunker_version
    doc.document_version = make_document_version(
        context, MINERU_PARSER_NAME, MINERU_PARSER_VERSION
    )

    # 页面尺寸（pt），用于把归一化 bbox 还原成真实坐标
    layout_pages = (
        ((middle.get("extensions") or {}).get("docvortex_layout") or {}).get("pages") or []
    )
    page_size: Dict[int, Tuple[float, float]] = {}
    for item in layout_pages:
        if not isinstance(item, dict):
            continue
        try:
            page_size[int(item["page_idx"])] = (
                float(item.get("width_pt") or 0.0),
                float(item.get("height_pt") or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            continue

    headings = _HeadingStack()
    noise_counts: Dict[str, int] = {}
    figure_count = 0
    table_count = 0

    for raw_page in middle.get("pages") or []:
        try:
            page_idx = int(raw_page.get("page_idx") or 0)
        except (TypeError, ValueError):
            page_idx = 0
        page_no = page_idx + 1                       # MinerU 0-based → DocIR 1-based
        width_pt, height_pt = page_size.get(page_idx, (0.0, 0.0))
        blocks: List[DocBlock] = []
        page_meta: Dict[str, Any] = {"mineru_page_idx": page_idx}

        for raw_block in raw_page.get("blocks") or []:
            btype = str(raw_block.get("type") or "")
            bbox = _norm_to_pt(raw_block.get("bbox"), width_pt, height_pt) or None
            block_index = raw_block.get("index")
            locator_base = f"page/{page_no:06d}/mineru/{int(block_index or 0):06d}"

            if btype in NOISE_TYPES:
                noise_counts[btype] = noise_counts.get(btype, 0) + 1
                continue

            if btype in HEADING_TYPES:
                text = _segment_text(raw_block.get("content"))
                if not text:
                    continue
                section_path = headings.push(int(raw_block.get("level") or 1), text)
                locator = f"{locator_base}/heading"
                blocks.append(
                    DocBlock(
                        block_id=make_block_id(doc.document_version, locator, "heading"),
                        block_type="heading",
                        text=text,
                        page=page_no,
                        bbox=bbox,
                        layout_type="heading",
                        section_path=section_path,
                        structural_locator=locator,
                        element_index=int(block_index or 0),
                        meta={
                            "mineru_type": btype,
                            "mineru_level": raw_block.get("level"),
                        },
                    )
                )
                continue

            if btype in TABLE_TYPES:
                segments = raw_block.get("content")
                table_seg, table_caption, table_unit = split_table_segments(segments)
                if table_seg is None:
                    table_seg = (segments or [{}])[0]
                html = str(table_seg.get("content") or "")
                rows, spans = html_table_to_rows(html)
                if not rows:
                    continue
                image_ref = table_seg.get("image_path")
                locator = f"{locator_base}/table"
                # 关键：把 MinerU 的表标题与单位交给 normalize_table，
                # 它会把「[表格] 标题 / [单位] xxx」重复写进每个行组，
                # 让检索单元自带单位上下文（金融数据缺单位就不可用）。
                sub_blocks = normalize_table(
                    rows,
                    document_version=doc.document_version,
                    locator=locator,
                    section_path=headings.path or None,
                    caption=table_caption,
                    unit=table_unit,
                    page=page_no,
                )
                md = rows_to_markdown(rows)
                for i, block in enumerate(sub_blocks):
                    block.bbox = bbox
                    block.table_md = md
                    block.source_region_id = f"mineru_table_{block_index}"
                    block.meta.update(
                        {
                            "mineru_type": "table",
                            "mineru_image_ref": image_ref,
                            "mineru_caption": table_caption,
                            "mineru_unit": table_unit,
                            "cell_spans": spans if i == 0 else None,
                        }
                    )
                    if image_ref:
                        block.meta["figure_ref"] = image_ref
                blocks.extend(sub_blocks)
                table_count += 1
                continue

            if btype in FIGURE_TYPES:
                segments = raw_block.get("content")
                body, fig_caption, fig_note = split_figure_segments(segments)
                if body is None:
                    body = (segments or [{}])[0]
                image_ref = body.get("image_path")
                # 图注优先用 caption 段，缺失时退回 footnote（多为 "资料来源:…"）
                caption = fig_caption or fig_note or _segment_text(segments)
                fig_text = str(body.get("content") or "").strip()
                # 图表块必须带可检索文本，否则会被 DocIR 质量门禁判为 empty_block，
                # 也会在检索单元里变成空块。组合顺序与 tables.serialize_group 保持一致。
                text_parts: List[str] = []
                if headings.path:
                    text_parts.append(f"[章节] {headings.path}")
                if caption:
                    text_parts.append(f"[图] {caption}")
                if fig_text:
                    text_parts.append(fig_text)
                locator = f"{locator_base}/figure"
                figure_count += 1
                blocks.append(
                    DocBlock(
                        block_id=make_block_id(doc.document_version, locator, "figure"),
                        block_type="figure",
                        text="\n".join(text_parts),
                        page=page_no,
                        bbox=bbox,
                        layout_type="figure",
                        caption=caption or None,
                        section_path=headings.path or None,
                        structural_locator=locator,
                        element_index=int(block_index or 0),
                        meta={
                            "mineru_type": btype,
                            "figure_kind": "chart" if btype == "chart" else "image",
                            "figure_ref": image_ref,
                            "figure_text": fig_text,
                        },
                    )
                )
                continue

            # 其余按正文处理（text / index / 未知类型）
            text = _segment_text(raw_block.get("content"))
            if not text:
                continue
            locator = f"{locator_base}/text"
            blocks.append(
                DocBlock(
                    block_id=make_block_id(doc.document_version, locator, "paragraph"),
                    block_type="paragraph",
                    text=text,
                    page=page_no,
                    bbox=bbox,
                    layout_type="paragraph",
                    section_path=headings.path or None,
                    structural_locator=locator,
                    element_index=int(block_index or 0),
                    meta={"mineru_type": btype},
                )
            )

        if render_page_images:
            ref = _render_page_image(file_path, page_idx, parsed_dir, page_image_dpi)
            if ref:
                page_meta["image_ref"] = ref

        if blocks or page_meta.get("image_ref"):
            doc.add_page(
                DocPage(
                    page_no=page_no,
                    width=int(width_pt) or None,
                    height=int(height_pt) or None,
                    blocks=blocks,
                    meta=page_meta,
                )
            )

    if doc.total_pages == 0:
        raise MineruUnavailable(f"MinerU 转换后没有任何页面: {file_path}")

    doc.meta["mineru_noise_counts"] = noise_counts
    doc.meta["mineru_table_count"] = table_count
    doc.meta["mineru_figure_count"] = figure_count
    return doc


def _render_page_image(
    pdf_path: Path, page_idx: int, parsed_dir: Path, dpi: int
) -> Optional[str]:
    """用 PyMuPDF 渲染整页图，供视觉/页级检索通道使用。

    旧链路的页图路径全部指向已不存在的 F 盘绝对路径，导致图像通道 100% 失效；
    这里改为在解析阶段就地渲染，并把相对路径写进 `page.meta["image_ref"]`。
    """
    try:
        import fitz

        rel = f"images/page_{page_idx}_full.png"
        target = parsed_dir / rel
        if target.is_file():
            return rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(pdf_path) as pdf:
            if page_idx >= pdf.page_count:
                return None
            pix = pdf[page_idx].get_pixmap(dpi=dpi, alpha=False)
            pix.save(str(target))
        return rel
    except Exception:  # noqa: BLE001 — 渲染失败不应阻断解析
        return None


# ---------------------------------------------------------------------------
# 调用 MinerU CLI
# ---------------------------------------------------------------------------

def run_mineru_cli(
    file_path: Path,
    out_dir: Path,
    *,
    tier: str = MINERU_TIER_DEFAULT,
    ocr_mode: str = "auto",
    timeout: int = 3600,
    page_range: Optional[str] = None,
) -> Path:
    """调用 `mineru-kit parse` 并用 zip 产出全部文件，返回 extracted 目录。"""
    if shutil.which("mineru-kit") is None:
        raise MineruUnavailable("未找到 mineru-kit 可执行文件")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mineru-kit", "parse", str(file_path),
        "-o", str(out_dir),
        "--tier", tier,
        "--format", "zip",
        "--ocr-mode", ocr_mode,
    ]
    if page_range:
        cmd += ["-p", page_range]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise MineruUnavailable(f"MinerU 超时（{timeout}s）: {file_path}") from exc
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or ""))[-800:]
        raise MineruUnavailable(f"MinerU 退出码 {proc.returncode}: {tail}")

    extracted = out_dir / "extracted"
    for zp in out_dir.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(extracted)
    if not extracted.is_dir():
        raise MineruUnavailable(f"MinerU 未产出 zip: {out_dir}")
    return extracted


def merge_shard_middles(shard_dirs: Sequence[Path], target_dir: Path) -> Path:
    """把分片解析的产物合并成一份完整 MiddleJson（按原始 page_idx 排序）。

    MinerU 用 `-p a-b` 分片时，`pages[].page_idx` 与图片命名都使用**原始页码**，
    因此合并只需按 page_idx 归并，无需做偏移；图片名天然不冲突。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    merged_pages: Dict[int, Dict[str, Any]] = {}
    merged_layout: Dict[int, Dict[str, Any]] = {}
    base: Optional[Dict[str, Any]] = None
    markdown_parts: List[str] = []

    for shard in shard_dirs:
        mid = next(Path(shard).rglob("middle_json.json"), None)
        if mid is None:
            continue
        data = json.loads(mid.read_text(encoding="utf-8"))
        if base is None:
            base = {
                "metadata": data.get("metadata"),
                "schema": data.get("schema"),
                "schema_version": data.get("schema_version"),
                "extensions": {"mineru": (data.get("extensions") or {}).get("mineru")},
            }
        for page in data.get("pages") or []:
            try:
                idx = int(page.get("page_idx") or 0)
            except (TypeError, ValueError):
                continue
            merged_pages[idx] = page
        layout_pages = (
            ((data.get("extensions") or {}).get("docvortex_layout") or {}).get("pages") or []
        )
        for item in layout_pages:
            try:
                merged_layout[int(item.get("page_idx") or 0)] = item
            except (TypeError, ValueError):
                continue

        md = mid.parent / "markdown.md"
        if md.is_file():
            markdown_parts.append(md.read_text(encoding="utf-8").strip())

        # 搬图片：文件名含原始页码，不会冲突
        for img in (mid.parent / "images").glob("*") if (mid.parent / "images").is_dir() else []:
            dest = images_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    if base is None:
        raise MineruUnavailable("分片合并失败：没有任何 shard 产出 middle_json")

    base["pages"] = [merged_pages[k] for k in sorted(merged_pages)]
    base.setdefault("extensions", {})["docvortex_layout"] = {
        "version": 1,
        "pages": [merged_layout[k] for k in sorted(merged_layout)],
    }
    base["is_full_document"] = True
    base["extensions"]["mineru_shards"] = len(shard_dirs)

    (target_dir / "middle_json.json").write_text(
        json.dumps(base, ensure_ascii=False), encoding="utf-8"
    )
    if markdown_parts:
        (target_dir / "markdown.md").write_text(
            "\n\n".join(markdown_parts), encoding="utf-8"
        )
    return target_dir


def run_mineru_sharded(
    file_path: Path,
    out_dir: Path,
    *,
    total_pages: int,
    shard_size: int = 30,
    tier: str = MINERU_TIER_DEFAULT,
    ocr_mode: str = "auto",
    timeout: int = 5400,
    progress: Optional[Any] = None,
) -> Path:
    """大文档分片解析：逐片调用 MinerU，再按原始页码合并。

    动机：248 页扫描件实测 34.5 s/页，单进程跑完整份会因耗时/内存累积而崩溃
    （2 并发与单并发都复现）。分片后每片独立进程，内存可回收。
    """
    shard_dirs: List[Path] = []
    ranges = [
        (start, min(start + shard_size - 1, total_pages))
        for start in range(1, total_pages + 1, shard_size)
    ]
    for index, (start, end) in enumerate(ranges, start=1):
        shard_out = out_dir / f"shard_{start:05d}_{end:05d}"
        extracted = shard_out / "extracted"
        if not (extracted / "middle_json.json").is_file():
            # MinerU 可能把 middle_json 放在更深的子目录，先探测
            if not any(extracted.rglob("middle_json.json")):
                if progress:
                    progress(f"  shard {index}/{len(ranges)} p{start}-{end} …")
                run_mineru_cli(
                    file_path, shard_out,
                    tier=tier, ocr_mode=ocr_mode, timeout=timeout,
                    page_range=f"{start}-{end}",
                )
        shard_dirs.append(extracted if (extracted / "middle_json.json").is_file()
                          else shard_out)
    return merge_shard_middles(shard_dirs, out_dir / "extracted")


def parse_with_mineru(
    file_path: Path,
    parsed_dir: Path,
    context: ParseContext,
    *,
    tier: str = MINERU_TIER_DEFAULT,
    ocr_mode: str = "auto",
    render_page_images: bool = True,
    total_pages: int = 0,
    shard_pages: int = 0,
) -> DocIR:
    """端到端：调用 MinerU → 读取 MiddleJson → DocIR v3。

    `total_pages > 0` 且 `shard_pages > 0` 且页数超过分片大小时，自动走分片解析
    （大文档单进程会因耗时/内存累积崩溃）。

    失败时抛 `MineruUnavailable`，由调用方（`process_document`）回退到
    `PDFDocumentParser` 并记录 `quality_flags=["mineru_fallback"]`。
    """
    if shard_pages > 0 and total_pages > shard_pages:
        extracted = run_mineru_sharded(
            file_path, parsed_dir,
            total_pages=total_pages, shard_size=shard_pages,
            tier=tier, ocr_mode=ocr_mode,
        )
    else:
        extracted = run_mineru_cli(file_path, parsed_dir, tier=tier, ocr_mode=ocr_mode)
    mid_path = next(extracted.rglob("middle_json.json"), None)
    if mid_path is None:
        raise MineruUnavailable(f"MinerU 产物缺少 middle_json.json: {parsed_dir}")
    middle = json.loads(mid_path.read_text(encoding="utf-8"))
    return middle_json_to_docir(
        middle,
        file_path=file_path,
        parsed_dir=extracted,
        context=context,
        tier=tier,
        render_page_images=render_page_images,
    )


def resolve_figure_ref(doc: DocIR, ref: Optional[str], *, search_root: Path) -> Optional[str]:
    """把 `figure_ref` 解析成可访问的路径。

    优先尝试 `search_root / ref`（解析产物根目录），再回退到绝对路径。
    """
    if not ref:
        return None
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate)
    resolved = search_root / ref
    if resolved.is_file():
        return str(resolved)
    return None
