import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.pipeline.docir import DocBlock, DocIR, DocPage
from src.pipeline.identity import ParseContext
from src.pipeline.parsers import PlainTextParser
from src.pipeline.quality import (
    QualityGateError,
    assess_document,
    assess_snapshot,
    process_document,
)


class ParseQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_duplicate_id_and_missing_locator_block_activation(self):
        doc = DocIR(
            doc_id="doc-1",
            source_path="rules.docx",
            source_format=".docx",
            document_version="dv-1",
            schema_version=2,
        )
        doc.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock("same", "paragraph", "甲", None),
                    DocBlock("same", "paragraph", "乙", None),
                ],
            )
        )

        report = assess_document(doc)

        self.assertEqual(report.status, "failed")
        self.assertEqual(report.duplicate_block_id_count, 1)
        self.assertEqual(report.missing_locator_count, 2)

    def test_uncertain_order_is_warning_not_hard_failure(self):
        doc = DocIR(
            doc_id="doc-1",
            source_path="rules.pdf",
            source_format=".pdf",
            document_version="dv-1",
            schema_version=2,
        )
        doc.add_page(
            DocPage(
                page_no=1,
                blocks=[
                    DocBlock(
                        "blk-1",
                        "paragraph",
                        "正文",
                        1,
                        structural_locator="page/000001/text/000001",
                        quality_flags=["uncertain_reading_order"],
                    )
                ],
            )
        )

        report = assess_document(doc)

        self.assertEqual(report.status, "passed_with_warnings")
        self.assertEqual(report.uncertain_reading_order_count, 1)

    def test_table_row_group_without_data_fails(self):
        doc = DocIR(
            doc_id="doc-1",
            source_path="sheet.xlsx",
            source_format=".xlsx",
            document_version="dv-1",
            schema_version=2,
        )
        doc.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock(
                        "blk-1",
                        "table_rows",
                        "",
                        None,
                        structural_locator="sheet/S1/rows/000001-000001",
                        table_headers=["项目"],
                    )
                ],
            )
        )

        report = assess_document(doc)

        self.assertEqual(report.status, "failed")
        self.assertIn("table_row_group_invalid", report.failure_reasons)

    def test_empty_document_fails(self):
        doc = DocIR(
            doc_id="doc-empty",
            source_path="empty.txt",
            source_format=".txt",
            document_version="dv-empty",
            schema_version=2,
        )

        self.assertEqual(assess_document(doc).status, "failed")

    def test_snapshot_report_aggregates_hard_failures(self):
        bad = DocIR(
            doc_id="doc-bad",
            source_path="bad.txt",
            source_format=".txt",
            document_version="dv-bad",
            schema_version=2,
        )
        good = DocIR(
            doc_id="doc-good",
            source_path="good.txt",
            source_format=".txt",
            document_version="dv-good",
            schema_version=2,
        )
        good.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock(
                        "blk-good",
                        "paragraph",
                        "正文",
                        None,
                        structural_locator="body/000000/paragraph",
                    )
                ],
            )
        )

        report = assess_snapshot([bad, good], snapshot_id="snap-1")

        self.assertEqual(report.document_count, 2)
        self.assertEqual(report.status, "failed")
        self.assertGreater(report.hard_failure_count, 0)

    def test_process_document_writes_versioned_artifacts(self):
        source = self.tmp_path / "rules.txt"
        source.write_text("规则正文", encoding="utf-8")
        context = ParseContext.for_file(source, "doc-txt", "cfg-v2", "chunk-v2")
        output = self.tmp_path / "candidate"

        processed = process_document(
            source,
            output,
            context,
            [PlainTextParser()],
        )

        self.assertTrue(processed.docir_path.is_file())
        self.assertTrue(processed.quality_path.is_file())
        doc = DocIR.from_dict(
            json.loads(processed.docir_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(doc.document_version, processed.document.document_version)
        self.assertEqual(processed.report.status, "passed")


if __name__ == "__main__":
    unittest.main()
