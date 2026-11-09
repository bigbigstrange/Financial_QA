"""chunker-v3 的行为契约测试。

覆盖三件必须成立的事：
  1. 切块结果确定性（可复现、可做缓存指纹）
  2. 语义完整性（不从句中落刀 + overlap 句对齐 + 表格/图表原子）
  3. 长度有界（不超 max_chars，且绝不丢内容）
"""
import unittest

from src.pipeline.chunker import (
    ChunkConfig,
    _ends_with_sentence_punct,
    _sentence_tail,
    build_chunks,
    chunk_stats,
    estimate_tokens,
)
from src.pipeline.docir import DocBlock
from src.pipeline.tables import normalize_table

CFG = ChunkConfig(
    target_chars=600,
    max_chars=900,
    min_chars=150,
    overlap_chars=96,
    chunker_version="chunker-v3-test",
)


def _b(bid: str, text: str, **kw) -> DocBlock:
    kw.setdefault("page", 1)
    kw.setdefault("structural_locator", f"test/{bid}")
    return DocBlock(block_id=bid, block_type=kw.pop("block_type", "paragraph"),
                    text=text, **kw)


def _sentence(marker: str, n: int) -> str:
    """生成恰好 n 字符、以句号结尾、前缀可区分的句子。"""
    body = (marker * ((n // len(marker)) + 1))[:n]
    return body[:-1] + "。"


def _paragraph(n_sent: int, sent_len: int, tag: str) -> str:
    return "".join(_sentence(f"{tag}{i}", sent_len) for i in range(n_sent))


class DeterminismTests(unittest.TestCase):
    def test_chunk_ids_and_text_are_deterministic(self):
        blocks = [
            _b("b1", _paragraph(2, 60, "甲")),
            _b("b2", _paragraph(2, 60, "乙")),
        ]
        first = build_chunks(blocks, document_version="dv-1", config=CFG)
        second = build_chunks(blocks, document_version="dv-1", config=CFG)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        self.assertEqual([c.text for c in first], [c.text for c in second])

    def test_different_document_version_changes_ids(self):
        blocks = [_b("b1", _paragraph(2, 60, "甲"))]
        a = build_chunks(blocks, document_version="dv-1", config=CFG)
        b = build_chunks(blocks, document_version="dv-2", config=CFG)
        self.assertNotEqual(a[0].chunk_id, b[0].chunk_id)

    def test_empty_input_returns_empty(self):
        self.assertEqual(build_chunks([], document_version="dv-1", config=CFG), [])

    def test_whitespace_only_blocks_are_dropped(self):
        blocks = [_b("b1", "   \n  "), _b("b2", _paragraph(1, 100, "甲"))]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_ids, ["b2"])


class AggregationTests(unittest.TestCase):
    def test_short_paragraphs_merge_into_one_chunk(self):
        """4 个 100 字段落（合计 400 < target 600）应聚合成 1 个 chunk，而不是 4 个碎片。"""
        blocks = [
            _b(f"b{i}", _paragraph(1, 100, "甲乙丙丁"[i])) for i in range(4)
        ]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].block_ids), 4)
        self.assertGreaterEqual(chunks[0].char_len, 400)

    def test_too_short_group_merges_with_previous(self):
        """前一段很长 + 后一段很短时，短段应并入前一个 chunk 而不是单独成块。"""
        blocks = [
            _b("b1", _paragraph(7, 100, "甲")),   # 700 字
            _b("b2", _paragraph(1, 80, "乙")),    # 80 字 < min_chars
        ]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(chunks[0].char_len, CFG.max_chars)

    def test_heading_opens_new_group_and_prefixes_body(self):
        heading = _b("h1", "一、业务概述", block_type="heading")
        blocks = [
            heading,
            _b("b1", _paragraph(3, 100, "甲")),
            _b("b2", _paragraph(3, 100, "乙")),
            _b("b3", _paragraph(3, 100, "丙")),
        ]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        self.assertTrue(chunks[0].text.startswith("一、业务概述"))
        self.assertIn("h1", chunks[0].block_ids)


class SentenceBoundaryTests(unittest.TestCase):
    def test_no_chunk_ends_mid_sentence(self):
        """核心验收指标：句子级内容不得从句中切断。"""
        blocks = [
            _b("b1", _paragraph(3, 120, "甲")),
            _b("b2", _paragraph(3, 120, "乙")),
            _b("b3", _paragraph(3, 120, "丙")),
        ]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        for chunk in chunks:
            self.assertNotIn(
                "not_sentence_end", chunk.quality_flags,
                msg=f"chunk {chunk.chunk_id} 未在句末结束: ...{chunk.text[-30:]!r}",
            )
            self.assertTrue(_ends_with_sentence_punct(chunk.text))

    def test_oversized_paragraph_is_split_at_sentence_not_hard_cut(self):
        block = _b("b1", _paragraph(30, 100, "甲"))  # 3000 字
        chunks = build_chunks([block], document_version="dv-1", config=CFG)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertNotIn("hard_split", chunk.quality_flags)
            self.assertTrue(_ends_with_sentence_punct(chunk.text))

    def test_unterminated_text_is_flagged_hard_split_and_still_bounded(self):
        block = _b("b1", "啊" * 2000)  # 完全没有句末标点
        chunks = build_chunks([block], document_version="dv-1", config=CFG)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertIn("hard_split", chunk.quality_flags)
            self.assertLessEqual(chunk.char_len, CFG.max_chars)
        # 关键：内容不能丢
        joined = "".join(c.text.replace("\n", "") for c in chunks)
        self.assertGreaterEqual(len(joined), 2000)


class OverlapTests(unittest.TestCase):
    def test_overlap_prefix_is_sentence_aligned(self):
        p1 = _paragraph(8, 100, "甲")   # 800 字
        p2 = _paragraph(3, 100, "乙")   # 300 字
        chunks = build_chunks(
            [_b("b1", p1), _b("b2", p2)], document_version="dv-1", config=CFG
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].overlap_prev_chars, 0)
        self.assertGreater(chunks[1].overlap_prev_chars, 0)
        # 重叠前缀必须是上一段的最后一个「完整句子」
        self.assertTrue(chunks[1].text.startswith(_sentence("甲7", 100)))
        # 而且不能被截断成半句
        prefix = chunks[1].text[: chunks[1].overlap_prev_chars]
        self.assertTrue(_ends_with_sentence_punct(prefix))

    def test_overlap_anchor_stays_within_previous_body(self):
        """重叠必须锚定在「上一段正文」内部，不能与前缀叠加造成链式增长。

        该性质由 `min_chars >= overlap_chars` 保证：正文不短于 min_chars 时，
        `_sentence_tail(body, overlap)` 永远落在 body 内部。
        """
        p1 = _paragraph(7, 100, "甲")   # 700 字
        p2 = _paragraph(7, 100, "乙")
        p3 = _paragraph(7, 100, "丙")
        chunks = build_chunks(
            [_b("b1", p1), _b("b2", p2), _b("b3", p3)],
            document_version="dv-1",
            config=CFG,
        )
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[1].text.startswith(_sentence("甲6", 100)))
        self.assertTrue(chunks[2].text.startswith(_sentence("乙6", 100)))
        for i in (1, 2):
            prefix = chunks[i].text[: chunks[i].overlap_prev_chars].rstrip("\n")
            self.assertGreater(len(prefix), 0)
            # 前缀是上一个 chunk 文本的结尾（因此也在其正文内部）
            self.assertTrue(chunks[i - 1].text.endswith(prefix))
            self.assertLessEqual(len(prefix), CFG.overlap_chars * 1.5)
        # 第三个 chunk 的重叠里不得掺入第一段的内容
        self.assertNotIn("甲", chunks[2].text[: chunks[2].overlap_prev_chars])

    def test_overlap_dropped_when_no_room_left(self):
        """单段就接近 max_chars 时放不下完整句子 → 放弃 overlap，而不是塞半句话。"""
        p1 = _paragraph(8, 100, "甲")   # 800 字，已接近 max_chars=900
        p2 = _paragraph(8, 100, "乙")
        chunks = build_chunks(
            [_b("b1", p1), _b("b2", p2)], document_version="dv-1", config=CFG
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1].overlap_prev_chars, 0)
        # 宁可没有 overlap，也必须保持句末完整与长度有界
        self.assertTrue(_ends_with_sentence_punct(chunks[1].text))
        self.assertLessEqual(chunks[1].char_len, CFG.max_chars)

    def test_chunk_length_never_exceeds_max(self):
        blocks = [
            _b(f"b{i}", _paragraph(4, 90, "甲乙丙丁"[i % 4])) for i in range(12)
        ]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        for chunk in chunks:
            self.assertLessEqual(
                chunk.char_len, CFG.max_chars,
                msg=f"{chunk.chunk_id} 超长 {chunk.char_len}",
            )

    def test_sentence_tail_is_bounded(self):
        text = _paragraph(8, 100, "甲")
        tail = _sentence_tail(text, 96)
        self.assertGreater(len(tail), 0)
        self.assertLessEqual(len(tail), 96 * 1.5)
        self.assertTrue(text.endswith(tail))


class TableAndFigureTests(unittest.TestCase):
    def _table_blocks(self):
        rows = [["期限", "金额"]] + [[f"{i}年", str(i * 10)] for i in range(1, 46)]
        return normalize_table(
            rows,
            document_version="dv-t",
            locator="page/000001/table/000001",
            section_path="流动性风险",
            caption="期限结构",
            unit="万元",
            max_rows=20,
        )

    def test_table_row_groups_are_atomic(self):
        table_blocks = self._table_blocks()
        parent, *groups = table_blocks
        chunks = build_chunks(
            [_b("p1", _paragraph(3, 100, "甲")), *table_blocks,
             _b("p2", _paragraph(3, 100, "乙"))],
            document_version="dv-t",
            config=CFG,
        )
        table_chunks = [c for c in chunks if c.block_type == "table"]
        self.assertEqual(len(table_chunks), len(groups))
        for chunk in table_chunks:
            self.assertEqual(len(chunk.block_ids), 1)          # 不与正文合并
            self.assertEqual(chunk.overlap_prev_chars, 0)      # 不接收 overlap 前缀
            self.assertEqual(chunk.section_path, "流动性风险")

    def test_table_parent_is_not_indexed_when_row_groups_exist(self):
        table_blocks = self._table_blocks()
        parent = table_blocks[0]
        chunks = build_chunks(
            table_blocks, document_version="dv-t", config=CFG
        )
        indexed = {bid for c in chunks for bid in c.block_ids}
        self.assertNotIn(parent.block_id, indexed)

    def test_table_chunk_text_repeats_headers_unit_and_section(self):
        chunks = build_chunks(
            self._table_blocks(), document_version="dv-t", config=CFG
        )
        for chunk in chunks:
            self.assertIn("[单位] 万元", chunk.text)
            self.assertIn("[列] 期限 | 金额", chunk.text)
            self.assertIn("[章节] 流动性风险", chunk.text)

    def test_table_text_rebuilt_when_block_text_empty(self):
        """旧知识库里表格块 text 为空 —— 必须能从结构化字段重建，否则检索不到。"""
        block = DocBlock(
            block_id="tbl1",
            block_type="table_rows",
            text="",
            section_path="资本充足率",
            unit="%",
            table_headers=["项目", "2023年"],
            table_rows=[["核心一级资本充足率", "9.8"]],
            structural_locator="page/000001/table/000001/rows/000002-000002",
        )
        chunks = build_chunks([block], document_version="dv-t", config=CFG)
        self.assertEqual(len(chunks), 1)
        self.assertIn("核心一级资本充足率", chunks[0].text)
        self.assertIn("[单位] %", chunks[0].text)

    def test_figure_content_is_merged_with_neighbouring_prose(self):
        """图表不再独立成组：图注/图内文字应与相邻正文合并，且内容不丢失。"""
        figure = DocBlock(
            block_id="fig1",
            block_type="figure",
            text="",
            caption="图 1 同业资产规模变化",
            section_path="流动性风险",
            bbox=[10.0, 20.0, 300.0, 400.0],
            structural_locator="page/000005/chart/000001",
            meta={"figure_text": "Y轴 0-5万亿，X轴 2009-2016"},
        )
        chunks = build_chunks(
            [_b("p1", _paragraph(2, 100, "甲")), figure,
             _b("p2", _paragraph(2, 100, "乙"))],
            document_version="dv-t",
            config=CFG,
        )
        merged = "\n".join(c.text for c in chunks)
        self.assertIn("图 1 同业资产规模变化", merged)
        self.assertIn("[章节] 流动性风险", merged)
        self.assertIn("Y轴 0-5万亿", merged)
        # 合并后仍应是可追溯的：figure 的 block_id 出现在某个 chunk 里
        self.assertTrue(any("fig1" in c.block_ids for c in chunks))

    def test_lone_figure_becomes_figure_chunk(self):
        figure = DocBlock(
            block_id="fig1",
            block_type="figure",
            text="",
            caption="图 2 资产质量趋势",
            structural_locator="page/000009/chart/000002",
            meta={"figure_text": "不良率 1.32%"},
        )
        chunks = build_chunks([figure], document_version="dv-t", config=CFG)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_type, "figure")
        self.assertIn("图 2 资产质量趋势", chunks[0].text)
        self.assertIn("不良率 1.32%", chunks[0].text)

    def test_mixed_chunk_is_labelled_paragraph_for_clean_metrics(self):
        """图表与正文混合的 chunk 按正文归类，避免污染图表类指标。"""
        figure = DocBlock(
            block_id="fig1", block_type="figure", text="",
            caption="图 3 净息差",
            structural_locator="page/000010/chart/000003",
        )
        chunks = build_chunks(
            [_b("p1", _paragraph(2, 100, "甲")), figure],
            document_version="dv-t",
            config=CFG,
        )
        self.assertEqual(chunks[0].block_type, "paragraph")


class TraceabilityTests(unittest.TestCase):
    def test_chunk_keeps_page_range_and_bbox_union(self):
        b1 = _b("b1", _paragraph(2, 100, "甲"), page=3, bbox=[10.0, 10.0, 200.0, 50.0])
        b2 = _b("b2", _paragraph(2, 100, "乙"), page=4, bbox=[20.0, 60.0, 300.0, 120.0])
        chunks = build_chunks([b1, b2], document_version="dv-1", config=CFG)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_start, 3)
        self.assertEqual(chunks[0].page_end, 4)
        self.assertEqual(chunks[0].bbox_union, [10.0, 10.0, 300.0, 120.0])

    def test_block_ids_are_traceable(self):
        blocks = [_b(f"b{i}", _paragraph(2, 100, "甲乙丙丁"[i])) for i in range(4)]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        indexed = [bid for c in chunks for bid in c.block_ids]
        self.assertEqual(sorted(indexed), ["b0", "b1", "b2", "b3"])

    def test_to_retrieval_item_contract(self):
        chunk = build_chunks(
            [_b("b1", _paragraph(2, 100, "甲"))], document_version="dv-1", config=CFG
        )[0]
        item = chunk.to_retrieval_item(
            source_doc="某银行2023年报.pdf",
            document_id="doc-1",
            document_version="dv-1",
            snapshot_id="snap-1",
        )
        for key in (
            "block_id", "chunk_id", "content", "source_doc", "page",
            "section_path", "type", "score",
        ):
            self.assertIn(key, item)
        self.assertEqual(item["block_id"], chunk.chunk_id)
        self.assertEqual(item["snapshot_id"], "snap-1")


class StatsTests(unittest.TestCase):
    def test_estimate_tokens_prefers_cjk_density(self):
        self.assertGreater(estimate_tokens("中文" * 100), estimate_tokens("ab" * 100))

    def test_chunk_stats_reports_sentence_end_ratio(self):
        blocks = [
            _b("b1", _paragraph(3, 120, "甲")),
            _b("b2", _paragraph(3, 120, "乙")),
        ]
        chunks = build_chunks(blocks, document_version="dv-1", config=CFG)
        stats = chunk_stats(chunks)
        self.assertEqual(stats["chunk_count"], len(chunks))
        self.assertEqual(stats["chunk_sentence_end_ratio"], 1.0)
        self.assertEqual(stats["chunk_under_min_count"], 0)
        self.assertEqual(stats["chunk_hard_split_count"], 0)
        self.assertGreater(stats["chunk_len_p50"], 0)

    def test_chunk_stats_on_empty_input(self):
        stats = chunk_stats([])
        self.assertEqual(stats["chunk_count"], 0)
        self.assertEqual(stats["chunk_sentence_end_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
