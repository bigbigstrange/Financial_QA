import unittest
from scripts.complete_full_parse import build_coverage
from src.pipeline.identity import artifact_key


class FullCoverageTests(unittest.TestCase):
    def test_indexed_duplicates_and_failures_all_accounted_for(self):
        records = [{"rel_path": p} for p in ("a.pdf", "b.doc", "c.pdf", "bad.doc", "missing.jpg")]
        key = artifact_key("a.pdf")
        kb = {"documents": [{"doc_key": key}, {"doc_key": "converted", "origin_rel_path": "b.doc"}],
              "stats": {"skipped": [{"doc_key": artifact_key("c.pdf"), "reason": "duplicate_of:" + key}]}}
        rows = build_coverage(records, kb, [], [{"rel_path": "bad.doc", "status": "timeout"}])
        self.assertEqual([r["status"] for r in rows], ["indexed", "indexed", "duplicate", "needs_review", "needs_review"])
        self.assertEqual(rows[3]["reason"], "conversion_timeout")
