import json
from pathlib import Path
import tempfile
import unittest

from src.pipeline.kb_builder import build_kb_v3, _rebase_refs
from src.pipeline.docir import DocIR, DocPage


class FullCorpusTextTests(unittest.TestCase):
    def test_plain_text_is_included_without_mineru_or_fake_page_number(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.txt").write_text("银行应按季度报送同业业务数据。" * 30, encoding="utf-8")
            subset = root / "subset.json"
            subset.write_text(json.dumps({"files": [{"rel_path": "source.txt", "ext": ".txt"}]}))
            result = build_kb_v3(root / "parsed", root / "kb", docs_root=root, subset_path=subset)
            self.assertEqual(result["document_count"], 1)
            chunks = [json.loads(line) for line in (root / "kb/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(chunks)
            self.assertIsNone(chunks[0]["page_start"])
            self.assertEqual(chunks[0]["source_doc"], "source.txt")

    def test_external_parsed_assets_resolve_when_building_candidate_kb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = DocIR(doc_id="d", source_path="a.pdf", source_format=".pdf")
            doc.add_page(DocPage(page_no=1, blocks=[], meta={"image_ref": "images/p1.png"}))
            _rebase_refs(doc, parse_dir=root / "old/extracted", kb_root=root / "new")
            self.assertEqual(Path(doc.pages[0].meta["image_ref"]), (root / "old/extracted/images/p1.png").resolve())
