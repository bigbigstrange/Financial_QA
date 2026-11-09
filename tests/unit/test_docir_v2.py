import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.pipeline.docir import DocBlock, DocIR, DocPage
from src.pipeline.identity import (
    ParseContext,
    make_block_id,
    make_document_version,
)


class DocIRV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_same_context_produces_same_version_and_block_id(self):
        source = self.tmp_path / "same.docx"
        source.write_bytes(b"same bytes")
        ctx = ParseContext.for_file(source, "doc-a", "cfg-1", "chunk-v2")
        version = make_document_version(ctx, "word_structured", "2")

        self.assertEqual(version, make_document_version(ctx, "word_structured", "2"))
        self.assertEqual(
            make_block_id(version, "body/000001/paragraph", "paragraph"),
            make_block_id(version, "body/000001/paragraph", "paragraph"),
        )

    def test_same_name_different_logical_documents_do_not_collide(self):
        source = self.tmp_path / "rules.docx"
        source.write_bytes(b"same bytes")
        one = ParseContext.for_file(source, "doc-a", "cfg-1", "chunk-v2")
        two = ParseContext.for_file(source, "doc-b", "cfg-1", "chunk-v2")

        self.assertNotEqual(
            make_document_version(one, "word_structured", "2"),
            make_document_version(two, "word_structured", "2"),
        )

    def test_word_location_does_not_require_fake_page(self):
        block = DocBlock(
            block_id="blk-a",
            block_type="paragraph",
            text="正文",
            page=None,
            structural_locator="body/000002/paragraph",
            element_index=2,
        )
        self.assertIsNone(block.page)
        self.assertEqual(block.structural_locator, "body/000002/paragraph")

    def test_v2_round_trip_preserves_nullable_page_and_locator(self):
        doc = DocIR(
            doc_id="doc-word",
            source_path="mixed.docx",
            source_format=".docx",
            document_version="dv-word",
            logical_document_id="uuid-word",
            schema_version=2,
        )
        doc.add_page(
            DocPage(
                page_no=None,
                blocks=[
                    DocBlock(
                        block_id="blk-1",
                        block_type="paragraph",
                        text="正文",
                        page=None,
                        structural_locator="body/000002/paragraph",
                    )
                ],
            )
        )

        restored = DocIR.from_dict(json.loads(json.dumps(doc.to_dict())))

        self.assertEqual(restored.schema_version, 2)
        self.assertEqual(restored.document_id, "uuid-word")
        self.assertIsNone(restored.pages[0].page_no)
        self.assertEqual(restored.pages[0].blocks[0].structural_locator, "body/000002/paragraph")

    def test_validate_v2_flags_duplicate_and_missing_locator(self):
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

        failures = doc.validate_v2()

        self.assertIn("duplicate_block_id", failures)
        self.assertEqual(failures.count("missing_structural_locator"), 1)


if __name__ == "__main__":
    unittest.main()
