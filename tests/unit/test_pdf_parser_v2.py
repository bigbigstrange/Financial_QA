import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from src.pipeline.identity import ParseContext
from src.pipeline.parsers import (
    PDFDocumentParser,
    _needs_ocr,
    _text_is_inside_table,
)


class PDFParserV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.parser = PDFDocumentParser()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _two_page_pdf(self) -> Path:
        path = self.tmp_path / "two-pages.pdf"
        pdf = fitz.open()
        for text in ("first page", "second page"):
            page = pdf.new_page()
            page.insert_text((72, 72), text)
        pdf.save(path)
        pdf.close()
        return path

    def test_pdf_keeps_real_page_bbox_and_deterministic_locator(self):
        path = self._two_page_pdf()
        context = ParseContext.for_file(path, "doc-pdf", "cfg-v2", "chunk-v2")
        doc = self.parser.parse(path, context=context)
        second = doc.pages[1].blocks[0]

        self.assertEqual(second.page, 2)
        self.assertEqual(len(second.bbox), 4)
        self.assertTrue(second.structural_locator.startswith("page/000002/text/"))
        self.assertEqual(doc.document_version.startswith("dv_"), True)

    def test_low_text_coverage_triggers_ocr_but_normal_text_does_not(self):
        self.assertTrue(_needs_ocr([], 100_000))
        self.assertFalse(
            _needs_ocr([(0, 0, 500, 100, "有效文本" * 30)], 100_000)
        )

    def test_text_overlap_with_table_is_marked_for_deduplication(self):
        self.assertTrue(
            _text_is_inside_table(
                "项目 金额",
                "| 项目 | 金额 |\n| --- | --- |\n| 现金 | 10 |",
            )
        )


if __name__ == "__main__":
    unittest.main()
