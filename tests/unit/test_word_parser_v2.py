import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from docx import Document

from src.pipeline.identity import ParseContext
from src.pipeline.parsers import WordDocumentParser


class WordParserV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.parser = WordDocumentParser()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_word_preserves_paragraph_table_paragraph_order(self):
        path = self.tmp_path / "mixed.docx"
        document = Document()
        document.add_heading("第一章", level=1)
        document.add_paragraph("表前")
        table = document.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "项目"
        table.rows[0].cells[1].text = "金额"
        table.rows[1].cells[0].text = "存款"
        table.rows[1].cells[1].text = "100"
        document.add_paragraph("表后")
        document.save(path)

        context = ParseContext.for_file(path, "doc-word", "cfg-v2", "chunk-v2")
        parsed = self.parser.parse(path, context=context)
        blocks = parsed.pages[0].blocks

        self.assertEqual(
            [b.block_type for b in blocks],
            ["heading", "paragraph", "table", "table_rows", "paragraph"],
        )
        self.assertTrue(all(b.page is None for b in blocks))
        self.assertEqual(blocks[-1].section_path, "第一章")
        self.assertTrue(blocks[3].structural_locator.startswith("body/000002/table/rows/"))

    def test_empty_paragraph_keeps_later_locator_index(self):
        path = self.tmp_path / "spacing.docx"
        document = Document()
        document.add_paragraph("")
        document.add_paragraph("正文")
        document.save(path)

        context = ParseContext.for_file(path, "doc-space", "cfg-v2", "chunk-v2")
        parsed = self.parser.parse(path, context=context)
        block = parsed.pages[0].blocks[0]

        self.assertEqual(block.block_type, "paragraph")
        self.assertEqual(block.structural_locator, "body/000001/paragraph")

    def test_nested_table_in_cell_gets_cell_locator(self):
        path = self.tmp_path / "nested.docx"
        document = Document()
        outer = document.add_table(rows=1, cols=1)
        outer.cell(0, 0).text = "外层"
        nested = outer.cell(0, 0).add_table(rows=2, cols=1)
        nested.cell(0, 0).text = "项目"
        nested.cell(1, 0).text = "嵌套"
        document.save(path)

        context = ParseContext.for_file(path, "doc-nested", "cfg-v2", "chunk-v2")
        parsed = self.parser.parse(path, context=context)
        blocks = parsed.pages[0].blocks

        self.assertTrue(
            any(
                b.structural_locator
                and "/cell/" in b.structural_locator
                and b.block_type == "table_rows"
                for b in blocks
            )
        )

    def test_legacy_call_is_explicitly_marked(self):
        path = self.tmp_path / "legacy.docx"
        document = Document()
        document.add_paragraph("正文")
        document.save(path)

        parsed = self.parser.parse(path)

        self.assertIn("legacy_identity", parsed.pages[0].blocks[0].quality_flags)


if __name__ == "__main__":
    unittest.main()
