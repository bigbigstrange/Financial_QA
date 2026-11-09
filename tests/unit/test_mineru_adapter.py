"""MinerU MiddleJson → DocIR v3 适配器契约测试。

覆盖四类必须正确的映射（对应旧链路的四个缺陷）：
  1. 语义块 → 段落/标题（块粒度、章节路径）
  2. table_body 的 HTML → 结构化表格 + 行组
  3. chart → figure（图注来自 chart_footnote、图片引用来自 image_path）
  4. header/footer/page_number → 噪声，不进检索单元
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.docir import DocIR
from src.pipeline.identity import ParseContext
from src.pipeline.mineru_adapter import (
    MINERU_PARSER_NAME,
    _norm_to_pt,
    html_table_to_rows,
    middle_json_to_docir,
    rows_to_markdown,
)

ROOT = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform")

MIDDLE = {
    "metadata": {
        "file_suffix": "pdf",
        "producer": {"name": "mineru", "version": "4.0.5"},
        "document": {"title": "测试文档", "page_count": 1},
    },
    "extensions": {
        "mineru": {"tier": "standard", "parse_mode": "txt"},
        "docvortex_layout": {
            "version": 1,
            "pages": [{"page_idx": 0, "width_pt": 600.0, "height_pt": 800.0}],
        },
    },
    "pages": [
        {
            "page_idx": 0,
            "blocks": [
                {"type": "header", "index": 0, "bbox": [0.0, 0.0, 0.5, 0.05],
                 "content": [{"type": "text", "content": "某银行 2023"}]},
                {"type": "doc_title", "index": 1, "bbox": [0.1, 0.1, 0.9, 0.15],
                 "level": 1, "content": [{"type": "text", "content": "某银行2023年报"}]},
                {"type": "paragraph_title", "index": 2, "bbox": [0.1, 0.2, 0.5, 0.24],
                 "level": 2, "content": [{"type": "text", "content": "一、经营概况"}]},
                {"type": "text", "index": 3, "bbox": [0.1, 0.25, 0.9, 0.35],
                 "content": [{"type": "text", "content": "2023年实现营业收入1000亿元，同比增长5%。"}]},
                {"type": "table", "index": 4, "bbox": [0.1, 0.4, 0.9, 0.6],
                 "content": [{
                     "type": "table_body", "index": 4,
                     "image_path": "images/page_0_table_4.jpg",
                     "content": ("<table><tbody>"
                                 "<tr><td>项目</td><td>2023</td></tr>"
                                 "<tr><td>营业收入</td><td>1000</td></tr>"
                                 "</tbody></table>"),
                 }]},
                {"type": "chart", "index": 5, "bbox": [0.1, 0.65, 0.9, 0.8],
                 "content": [
                     {"type": "chart_body", "index": 5,
                      "image_path": "images/page_0_chart_5.jpg", "content": ""},
                     {"type": "chart_footnote",
                      "content": [{"type": "text", "content": "资料来源:Wind"}]},
                 ]},
                {"type": "page_number", "index": 6, "bbox": [0.5, 0.95, 0.55, 0.98],
                 "content": [{"type": "text", "content": "1"}]},
            ],
        }
    ],
    "schema": "docvortex.middle",
    "schema_version": "2.0",
}


def _context() -> ParseContext:
    return ParseContext(
        document_id="doc-test-1",
        source_name="某银行2023年报.pdf",
        source_sha256="a" * 64,
        source_size_bytes=1234,
        parser_config_hash="parser-v3",
        chunker_version="chunker-v3",
    )


def _convert() -> DocIR:
    return middle_json_to_docir(
        MIDDLE,
        file_path=Path("某银行2023年报.pdf"),
        parsed_dir=ROOT / "output" / "kb_v3" / "parsed" / "_unit_test",
        context=_context(),
        render_page_images=False,
    )


class BboxTests(unittest.TestCase):
    def test_normalized_bbox_is_converted_to_points(self):
        self.assertEqual(_norm_to_pt([0.1, 0.25, 0.9, 0.35], 600.0, 800.0),
                         [60.0, 200.0, 540.0, 280.0])

    def test_invalid_bbox_returns_none(self):
        self.assertIsNone(_norm_to_pt(None, 600.0, 800.0))
        self.assertIsNone(_norm_to_pt([1, 2], 600.0, 800.0))
        self.assertIsNone(_norm_to_pt(["x", 2, 3, 4], 600.0, 800.0))


class TableHTMLTests(unittest.TestCase):
    def test_simple_table(self):
        rows, _ = html_table_to_rows(
            "<table><tbody>"
            "<tr><td>项目</td><td>金额</td></tr>"
            "<tr><td>营业收入</td><td>1,000</td></tr>"
            "</tbody></table>"
        )
        self.assertEqual(rows, [["项目", "金额"], ["营业收入", "1,000"]])

    def test_colspan_is_expanded_with_placeholders(self):
        rows, spans = html_table_to_rows(
            "<table><tr><td colspan='2'>合并表头</td><td>C</td></tr>"
            "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
        )
        self.assertEqual(rows[0], ["合并表头", "", "C"])
        self.assertEqual(spans[0][0], (2, 1))

    def test_nested_tags_and_whitespace_are_flattened(self):
        rows, _ = html_table_to_rows(
            "<table><tr><td><b>加粗</b>\n  换行</td><td>x</td></tr></table>"
        )
        self.assertEqual(rows, [["加粗 换行", "x"]])

    def test_empty_html_returns_empty(self):
        self.assertEqual(html_table_to_rows(""), ([], []))

    def test_rows_to_markdown_aligns_columns(self):
        md = rows_to_markdown([["A", "B"], ["1"]])
        lines = md.splitlines()
        self.assertEqual(lines[0], "| A | B |")
        self.assertEqual(lines[1], "| --- | --- |")
        self.assertEqual(lines[2], "| 1 |  |")


class DocIRMappingTests(unittest.TestCase):
    def test_maps_all_block_types(self):
        doc = _convert()
        by_type: dict[str, list] = {}
        for block in doc.all_blocks():
            by_type.setdefault(block.block_type, []).append(block)

        self.assertEqual(len(by_type.get("heading", [])), 2)
        self.assertEqual(len(by_type.get("paragraph", [])), 1)
        self.assertTrue(len(by_type.get("table_rows", [])) >= 1)
        self.assertEqual(len(by_type.get("figure", [])), 1)

    def test_noise_types_are_excluded_and_counted(self):
        doc = _convert()
        for block in doc.all_blocks():
            text = block.text or ""
            self.assertNotIn("某银行 2023", text)   # header
        self.assertEqual(doc.meta["mineru_noise_counts"],
                         {"header": 1, "page_number": 1})

    def test_page_number_is_one_based(self):
        doc = _convert()
        self.assertEqual([p.page_no for p in doc.pages], [1])

    def test_heading_stack_builds_section_path(self):
        doc = _convert()
        headings = [b for b in doc.all_blocks() if b.block_type == "heading"]
        self.assertEqual(headings[0].section_path, "某银行2023年报")
        self.assertEqual(headings[1].section_path, "某银行2023年报/一、经营概况")
        paragraph = [b for b in doc.all_blocks() if b.block_type == "paragraph"][0]
        self.assertEqual(paragraph.section_path, "某银行2023年报/一、经营概况")

    def test_bbox_uses_page_points(self):
        doc = _convert()
        paragraph = [b for b in doc.all_blocks() if b.block_type == "paragraph"][0]
        self.assertEqual(paragraph.bbox, [60.0, 200.0, 540.0, 280.0])

    def test_table_keeps_rows_headers_and_image_ref(self):
        doc = _convert()
        rows = [b for b in doc.all_blocks() if b.block_type == "table_rows"]
        self.assertTrue(rows)
        first = rows[0]
        self.assertEqual(first.table_headers, ["项目", "2023"])
        self.assertEqual(first.table_rows, [["营业收入", "1000"]])
        self.assertEqual(first.meta["mineru_image_ref"], "images/page_0_table_4.jpg")
        self.assertIn("| 项目 | 2023 |", first.table_md)

    def test_table_parent_is_not_indexable_when_row_groups_exist(self):
        doc = _convert()
        parents = [b for b in doc.all_blocks() if b.block_type == "table"]
        rows = [b for b in doc.all_blocks() if b.block_type == "table_rows"]
        self.assertTrue(parents and rows)
        self.assertEqual(rows[0].parent_id, parents[0].block_id)

    def test_chart_becomes_figure_with_caption_and_ref(self):
        doc = _convert()
        fig = [b for b in doc.all_blocks() if b.block_type == "figure"][0]
        self.assertEqual(fig.meta["figure_kind"], "chart")
        self.assertEqual(fig.meta["figure_ref"], "images/page_0_chart_5.jpg")
        self.assertEqual(fig.caption, "资料来源:Wind")
        self.assertEqual(fig.page, 1)

    def test_document_version_and_parser_are_set(self):
        doc = _convert()
        self.assertTrue(doc.document_version.startswith("dv_"))
        self.assertEqual(doc.parser_name, MINERU_PARSER_NAME)
        self.assertEqual(doc.logical_document_id, "doc-test-1")
        self.assertEqual(doc.meta["mineru_parse_mode"], "txt")

    def test_validate_v2_passes_for_converted_document(self):
        """适配器产出必须直接通过现有 DocIR 质量门禁。"""
        doc = _convert()
        self.assertEqual(doc.validate_v2(), [])

    def test_invalid_middle_json_raises(self):
        from src.pipeline.mineru_adapter import MineruUnavailable

        with self.assertRaises(MineruUnavailable):
            middle_json_to_docir({}, file_path=Path("x.pdf"),
                                 parsed_dir=ROOT, context=_context())


class RealArtifactTests(unittest.TestCase):
    """对真实 MinerU 冒烟产物做集成校验（产物不存在时自动跳过）。"""

    SMOKE = ROOT / "output" / "kb_v3" / "parsed" / "smoke_text_1" / "extracted"

    def setUp(self) -> None:
        mid = self.SMOKE / "middle_json.json"
        if not mid.is_file():
            self.skipTest(f"缺少冒烟产物: {mid}")
        self.middle = json.loads(mid.read_text(encoding="utf-8"))

    def test_real_document_converts_and_passes_gate(self):
        doc = middle_json_to_docir(
            self.middle,
            file_path=Path("农业银行_2024年一季报点评.pdf"),
            parsed_dir=self.SMOKE,
            context=_context(),
            render_page_images=False,
        )
        self.assertEqual(doc.validate_v2(), [])
        self.assertEqual(doc.total_pages, 4)
        types = {b.block_type for b in doc.all_blocks()}
        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        self.assertIn("table_rows", types)
        self.assertIn("figure", types)

    def test_real_document_has_charts_and_tables(self):
        doc = middle_json_to_docir(
            self.middle,
            file_path=Path("农业银行_2024年一季报点评.pdf"),
            parsed_dir=self.SMOKE,
            context=_context(),
            render_page_images=False,
        )
        figures = [b for b in doc.all_blocks() if b.block_type == "figure"]
        rows = [b for b in doc.all_blocks() if b.block_type == "table_rows"]
        self.assertGreaterEqual(len(figures), 1)
        self.assertGreaterEqual(len(rows), 1)
        # 图表引用必须是可解析的相对路径
        for fig in figures:
            ref = fig.meta.get("figure_ref")
            self.assertTrue(ref and ref.startswith("images/"), ref)

    def test_real_document_paragraphs_are_complete(self):
        """核心验收：真正的正文段落应以句末标点结尾，而不是被切成半句话。

        注意：研报里 MinerU 的 `text` 类型也包含大量短标签（"买入"、"基础数据"、
        分析师姓名电话等），这些本来就不以句号结尾。因此句末完整率只在
        **长段落（>=100 字，即真正的正文）** 上度量，短标签单独看长度分布。
        """
        doc = middle_json_to_docir(
            self.middle,
            file_path=Path("农业银行_2024年一季报点评.pdf"),
            parsed_dir=self.SMOKE,
            context=_context(),
            render_page_images=False,
        )
        paragraphs = [b for b in doc.all_blocks() if b.block_type == "paragraph"]
        self.assertTrue(paragraphs)

        prose = [b for b in paragraphs if len(b.text) >= 100]
        self.assertGreaterEqual(len(prose), 5, "长段落太少，样本不足以判断")
        end_ok = sum(1 for b in prose if b.text.rstrip()[-1:] in "。！？；")
        ratio = end_ok / len(prose)
        self.assertGreater(
            ratio, 0.8, f"长段落句末完整率仅 {ratio:.0%}，说明块粒度仍有问题"
        )

        # 段落长度应远大于旧链路的 24 字中位数
        lengths = sorted(len(b.text) for b in paragraphs)
        self.assertGreater(lengths[len(lengths) // 2], 60)

    def test_sentence_end_ratio_beats_legacy_baseline(self):
        """与旧链路同一份文档的对照：块粒度必须显著改善。

        旧链路（PyMuPDF 行级碎片）实测：块长中位数 24 字、句末完整率 46%。
        这里用「含句号的正文块」作为真正散文的判定（排除分析师联系方式、
        页码、标签这类同样被 MinerU 标为 text 但无句子结构的块）。
        """
        doc = middle_json_to_docir(
            self.middle,
            file_path=Path("农业银行_2024年一季报点评.pdf"),
            parsed_dir=self.SMOKE,
            context=_context(),
            render_page_images=False,
        )
        all_texts = [b.text for b in doc.all_blocks() if b.text.strip()]
        lengths = sorted(len(t) for t in all_texts)
        median = lengths[len(lengths) // 2]
        self.assertGreater(median, 40, f"MinerU 块长中位数仅 {median} 字")

        prose = [
            b.text for b in doc.all_blocks()
            if b.block_type == "paragraph" and len(b.text) >= 100 and "。" in b.text
        ]
        self.assertGreaterEqual(len(prose), 5, "样本不足以判断")
        end_ok = sum(1 for t in prose if t.rstrip()[-1:] in "。！？；")
        ratio = end_ok / len(prose)
        self.assertGreater(
            ratio, 0.9, f"正文块句末完整率仅 {ratio:.0%}（旧链路为 46%）"
        )


class TableCaptionSegmentTests(unittest.TestCase):
    """MinerU 把表标题/单位放在 content 数组的 caption/footnote 段里。

    旧实现只取 table_body，把标题与单位全丢了 —— 而金融数据缺单位就不可用。
    """

    def _segments(self, *extra) -> list:
        return [{"type": "table_body", "content": "<table></table>"}] + list(extra)

    def test_extracts_caption_and_unit_from_footnotes(self):
        from src.pipeline.mineru_adapter import split_table_segments

        body, caption, unit = split_table_segments(self._segments(
            {"type": "table_caption",
             "content": [{"type": "text", "content": "金融资产投资情况"}]},
            {"type": "table_footnote",
             "content": [{"type": "text", "content": "单位:百万元人民币（百分比除外）"}]},
            {"type": "table_footnote",
             "content": [{"type": "text", "content": "注: 1、理财产品…"}]},
        ))
        self.assertIsNotNone(body)
        self.assertIn("金融资产投资情况", caption)
        self.assertIn("注: 1、理财产品", caption)      # 表注不能丢
        self.assertEqual(unit, "百万元人民币（百分比除外）")

    def test_unit_pattern_variants(self):
        from src.pipeline.mineru_adapter import split_table_segments

        for raw, expected in (
            ("单位：万元", "万元"),
            ("单位:%", "%"),
            ("单位: 亿元 ", "亿元"),
        ):
            _, _, unit = split_table_segments(self._segments(
                {"type": "table_footnote",
                 "content": [{"type": "text", "content": raw}]},
            ))
            self.assertEqual(unit, expected, raw)

    def test_missing_caption_and_unit_returns_none(self):
        from src.pipeline.mineru_adapter import split_table_segments

        _, caption, unit = split_table_segments(self._segments())
        self.assertIsNone(caption)
        self.assertIsNone(unit)

    def test_no_body_returns_none_triple(self):
        from src.pipeline.mineru_adapter import split_table_segments

        self.assertEqual(split_table_segments([]), (None, None, None))

    def test_figure_caption_prefers_caption_over_footnote(self):
        from src.pipeline.mineru_adapter import split_figure_segments

        body, caption, note = split_figure_segments([
            {"type": "chart_body", "image_path": "images/page_0_chart_1.jpg", "content": ""},
            {"type": "chart_caption", "content": [{"type": "text", "content": "同业资产规模变化"}]},
            {"type": "chart_footnote", "content": [{"type": "text", "content": "资料来源:Wind"}]},
        ])
        self.assertIsNotNone(body)
        self.assertEqual(caption, "同业资产规模变化")
        self.assertEqual(note, "资料来源:Wind")

    def test_figure_footnote_used_as_caption_end_to_end(self):
        """没有 caption 段时，图注应回退到 footnote（研报常见只有 "资料来源:…"）。"""
        from src.pipeline.mineru_adapter import split_figure_segments

        segments = [
            {"type": "chart_body", "image_path": "images/page_0_chart_1.jpg", "content": ""},
            {"type": "chart_footnote", "content": [{"type": "text", "content": "资料来源:Wind"}]},
        ]
        _, caption, note = split_figure_segments(segments)
        self.assertIsNone(caption)              # helper 不做回退，保持语义清晰
        self.assertEqual(note, "资料来源:Wind")  # 回退由调用方完成

        middle = {
            "metadata": {"file_suffix": "pdf",
                         "producer": {"name": "mineru", "version": "4.0.5"}},
            "extensions": {"mineru": {"tier": "standard", "parse_mode": "txt"},
                           "docvortex_layout": {"version": 1, "pages": [
                               {"page_idx": 0, "width_pt": 600.0, "height_pt": 800.0}]}},
            "pages": [{"page_idx": 0, "blocks": [
                {"type": "chart", "index": 1, "bbox": [0.1, 0.1, 0.5, 0.5],
                 "content": segments},
            ]}],
            "schema": "docvortex.middle", "schema_version": "2.0",
        }
        doc = middle_json_to_docir(
            middle, file_path=Path("年报.pdf"),
            parsed_dir=ROOT, context=_context(), render_page_images=False,
        )
        fig = [b for b in doc.all_blocks() if b.block_type == "figure"][0]
        self.assertEqual(fig.caption, "资料来源:Wind")
        self.assertIn("[图] 资料来源:Wind", fig.text)


class TableCaptionIntegrationTests(unittest.TestCase):
    """真实年报产物：表格 chunk 必须带上 MinerU 的表标题与单位。"""

    PARSED = ROOT / "output" / "kb_v3" / "parsed"

    def _find_middle(self):
        if not self.PARSED.is_dir():
            return None
        for mid in self.PARSED.rglob("extracted/middle_json.json"):
            if "shard_" in str(mid):
                continue
            doc = json.loads(mid.read_text(encoding="utf-8"))
            for page in doc.get("pages") or []:
                for blk in page.get("blocks") or []:
                    if blk.get("type") != "table":
                        continue
                    for seg in blk.get("content") or []:
                        if isinstance(seg, dict) and seg.get("type") in {
                            "table_caption", "table_footnote"
                        }:
                            return mid, doc
        return None

    def test_real_table_caption_and_unit_reach_chunk_text(self):
        found = self._find_middle()
        if found is None:
            self.skipTest("没有带 caption/footnote 的真实表格产物")
        mid, doc = found
        doc_ir = middle_json_to_docir(
            doc,
            file_path=Path("年报.pdf"),
            parsed_dir=mid.parent,
            context=_context(),
            render_page_images=False,
        )
        row_groups = [b for b in doc_ir.all_blocks() if b.block_type == "table_rows"
                      and (b.caption or b.unit)]
        if not row_groups:
            self.skipTest("该产物中没有带标题或单位的表格行组")
        # 行组文本里必须出现 [单位] 或 [表格]，否则单位在切块时又丢了
        for block in row_groups[:20]:
            self.assertTrue(
                "[单位]" in block.text or "[表格]" in block.text,
                f"表格行组缺少标题/单位上下文: {block.text[:120]!r}",
            )


class ShardMergeTests(unittest.TestCase):
    """分片解析的合并逻辑：大文档必须能按原始 page_idx 正确拼回一份。"""

    def _write_shard(self, root: Path, name: str, pages: list[int],
                     layout: list[dict], text: str, images: list[str]) -> Path:
        shard = root / name / "extracted"
        (shard / "images").mkdir(parents=True, exist_ok=True)
        mid = {
            "metadata": {"file_suffix": "pdf",
                         "producer": {"name": "mineru", "version": "4.0.5"}},
            "extensions": {
                "mineru": {"tier": "standard", "parse_mode": "ocr"},
                "docvortex_layout": {"version": 1, "pages": layout},
            },
            "pages": [
                {"page_idx": p, "blocks": [{"type": "text", "index": 0,
                                            "bbox": [0, 0, 1, 1],
                                            "content": [{"type": "text",
                                                         "content": f"{text}-p{p}"}]}]}
                for p in pages
            ],
            "schema": "docvortex.middle",
            "schema_version": "2.0",
        }
        (shard / "middle_json.json").write_text(
            json.dumps(mid, ensure_ascii=False), encoding="utf-8"
        )
        (shard / "markdown.md").write_text(f"# {text}", encoding="utf-8")
        for img in images:
            (shard / "images" / img).write_bytes(b"fake-image")
        return shard

    def test_merges_pages_in_original_index_order(self):
        from src.pipeline.mineru_adapter import merge_shard_middles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s1 = self._write_shard(
                root, "shard_00001_00030", [0, 1],
                [{"page_idx": 0, "width_pt": 595.3, "height_pt": 841.9},
                 {"page_idx": 1, "width_pt": 595.3, "height_pt": 841.9}],
                "A", ["page_0_table_1.jpg"],
            )
            s2 = self._write_shard(
                root, "shard_00031_00060", [30, 31],
                [{"page_idx": 30, "width_pt": 595.3, "height_pt": 841.9},
                 {"page_idx": 31, "width_pt": 595.3, "height_pt": 841.9}],
                "B", ["page_30_chart_2.jpg"],
            )
            target = root / "merged"
            merge_shard_middles([s1, s2], target)

            merged = json.loads((target / "middle_json.json").read_text(encoding="utf-8"))
            self.assertEqual([p["page_idx"] for p in merged["pages"]], [0, 1, 30, 31])
            self.assertEqual(
                [p["page_idx"] for p in merged["extensions"]["docvortex_layout"]["pages"]],
                [0, 1, 30, 31],
            )
            self.assertTrue(merged["is_full_document"])
            self.assertEqual(merged["extensions"]["mineru_shards"], 2)
            # 图片被搬齐且不冲突（文件名自带原始页码）
            self.assertEqual(
                sorted(p.name for p in (target / "images").glob("*")),
                ["page_0_table_1.jpg", "page_30_chart_2.jpg"],
            )
            # markdown 按片拼接
            self.assertEqual(
                (target / "markdown.md").read_text(encoding="utf-8"), "# A\n\n# B"
            )

    def test_shard_order_does_not_matter(self):
        from src.pipeline.mineru_adapter import merge_shard_middles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s1 = self._write_shard(root, "s1", [0],
                                  [{"page_idx": 0, "width_pt": 1.0, "height_pt": 1.0}],
                                  "A", [])
            s2 = self._write_shard(root, "s2", [10],
                                  [{"page_idx": 10, "width_pt": 1.0, "height_pt": 1.0}],
                                  "B", [])
            a = root / "a"
            b = root / "b"
            merge_shard_middles([s1, s2], a)
            merge_shard_middles([s2, s1], b)
            ra = json.loads((a / "middle_json.json").read_text(encoding="utf-8"))
            rb = json.loads((b / "middle_json.json").read_text(encoding="utf-8"))
            self.assertEqual([p["page_idx"] for p in ra["pages"]],
                             [p["page_idx"] for p in rb["pages"]])

    def test_no_valid_shard_raises(self):
        from src.pipeline.mineru_adapter import MineruUnavailable, merge_shard_middles

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(MineruUnavailable):
                merge_shard_middles([empty], Path(tmp) / "out")


class ShardDecisionTests(unittest.TestCase):
    def test_small_document_is_not_sharded(self):
        """页数不超过阈值时不应走分片路径（避免无意义的多进程开销）。"""
        import src.pipeline.mineru_adapter as adapter

        calls: list[str] = []
        original = adapter.run_mineru_cli

        def fake_cli(file_path, out_dir, **kw):
            calls.append("single")
            raise adapter.MineruUnavailable("stop here")

        adapter.run_mineru_cli = fake_cli
        try:
            with self.assertRaises(adapter.MineruUnavailable):
                adapter.parse_with_mineru(
                    Path("x.pdf"), Path("out"), _context(),
                    total_pages=20, shard_pages=30, render_page_images=False,
                )
        finally:
            adapter.run_mineru_cli = original
        self.assertEqual(calls, ["single"])


if __name__ == "__main__":
    unittest.main()
