import unittest
from src.pipeline.docir import DocIR, DocPage, DocBlock
from scripts.register_kb_snapshot import same_evidence_content


class RegisterSnapshotTests(unittest.TestCase):
    def doc(self, source, text):
        doc = DocIR(doc_id="doc", source_path=source, source_format=".txt", document_version="dv")
        doc.add_page(DocPage(page_no=None, blocks=[DocBlock("blk", "paragraph", text, None)]))
        return doc

    def test_relocation_preserves_version(self):
        self.assertTrue(same_evidence_content(self.doc("old/file.txt", "原文"), self.doc("new/file.txt", "原文")))

    def test_changed_text_cannot_reuse_version(self):
        self.assertFalse(same_evidence_content(self.doc("old/file.txt", "原文"), self.doc("new/file.txt", "变更")))
