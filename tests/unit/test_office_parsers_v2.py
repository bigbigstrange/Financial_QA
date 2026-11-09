import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook
from pptx import Presentation

from src.pipeline.identity import ParseContext
from src.pipeline.parsers import (
    ExcelDocumentParser,
    PlainTextParser,
    PowerPointDocumentParser,
    RtfDocumentParser,
)


class OfficeParserV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_excel_uses_sheet_rows_instead_of_fake_page(self):
        path = self.tmp_path / "sheet.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "资产负债表"
        sheet.append(["项目", "金额"])
        sheet.append(["现金", 10])
        workbook.save(path)

        context = ParseContext.for_file(path, "doc-xlsx", "cfg-v2", "chunk-v2")
        doc = ExcelDocumentParser().parse(path, context=context)
        child = next(
            block for block in doc.pages[0].blocks if block.block_type == "table_rows"
        )

        self.assertIsNone(child.page)
        self.assertTrue(child.structural_locator.startswith("sheet/资产负债表/rows/"))
        self.assertTrue(child.parent_id)

    def test_ppt_uses_slide_and_shape_locator(self):
        path = self.tmp_path / "slides.pptx"
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "风险说明"
        presentation.save(path)

        context = ParseContext.for_file(path, "doc-ppt", "cfg-v2", "chunk-v2")
        doc = PowerPointDocumentParser().parse(path, context=context)

        self.assertTrue(
            doc.pages[0].blocks[0].structural_locator.startswith(
                "slide/000001/shape/"
            )
        )
        self.assertEqual(doc.pages[0].blocks[0].page, 1)

    def test_excel_truncation_is_explicit(self):
        path = self.tmp_path / "wide.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "大表"
        sheet.append(["序号", "值"])
        for i in range(600):
            sheet.append([i, i * 2])
        workbook.save(path)

        context = ParseContext.for_file(path, "doc-wide", "cfg-v2", "chunk-v2")
        doc = ExcelDocumentParser().parse(path, context=context)
        flags = [flag for block in doc.pages[0].blocks for flag in block.quality_flags]

        self.assertIn("truncated_rows", flags)

    def test_rtf_uses_real_parser_and_legacy_layout_flag(self):
        path = self.tmp_path / "legacy.rtf"
        path.write_text(r"{\rtf1\ansi 正文段落}", encoding="utf-8")

        context = ParseContext.for_file(path, "doc-rtf", "cfg-v2", "chunk-v2")
        doc = RtfDocumentParser().parse(path, context=context)

        self.assertEqual(doc.parser_name, "rtf_striprtf")
        self.assertIn("legacy_layout", doc.pages[0].blocks[0].quality_flags)
        self.assertIsNone(doc.pages[0].blocks[0].page)

    def test_plain_text_has_no_fake_page(self):
        path = self.tmp_path / "rules.txt"
        path.write_text("规则正文", encoding="utf-8")
        context = ParseContext.for_file(path, "doc-txt", "cfg-v2", "chunk-v2")

        doc = PlainTextParser().parse(path, context=context)

        self.assertIsNone(doc.pages[0].page_no)
        self.assertEqual(
            doc.pages[0].blocks[0].structural_locator,
            "body/000000/paragraph",
        )


if __name__ == "__main__":
    unittest.main()
