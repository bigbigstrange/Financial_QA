import tempfile
import unittest
from pathlib import Path

from scripts.audit_parser_baseline import audit_knowledge_base, load_json


def valid_kb():
    return {
        "documents": [
            {
                "doc_id": "doc-1",
                "pages": [
                    {
                        "page_no": 1,
                        "blocks": [
                            {"block_id": "doc-1-p1-b1", "block_type": "paragraph", "text": "正文"},
                            {"block_id": "doc-1-p1-t1", "block_type": "table", "text": "", "table_md": "|A|"},
                        ],
                    }
                ],
            }
        ]
    }


class ParserBaselineTests(unittest.TestCase):
    def test_valid_kb_is_clean(self):
        report = audit_knowledge_base(valid_kb(), {"summary": {"failed_files": 0}})
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["blocks"], 2)
        self.assertEqual(report["counts"]["table_blocks"], 1)

    def test_duplicate_block_id_fails(self):
        kb = valid_kb()
        kb["documents"][0]["pages"][0]["blocks"].append(
            {"block_id": "doc-1-p1-b1", "block_type": "paragraph", "text": "重复"}
        )
        report = audit_knowledge_base(kb, {"summary": {"failed_files": 0}})
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"]["duplicate_block_ids"], ["doc-1-p1-b1"])

    def test_document_without_blocks_fails(self):
        report = audit_knowledge_base(
            {"documents": [{"doc_id": "empty", "pages": []}]},
            {"summary": {"failed_files": 0}},
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"]["documents_without_blocks"], ["empty"])

    def test_empty_knowledge_base_fails(self):
        report = audit_knowledge_base(
            {"documents": []},
            {"summary": {"failed_files": 0}},
        )

        self.assertFalse(report["passed"])
        self.assertTrue(report["issues"]["empty_knowledge_base"])

    def test_bad_json_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_json(path)


if __name__ == "__main__":
    unittest.main()
