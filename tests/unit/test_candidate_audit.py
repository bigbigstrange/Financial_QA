import json
from pathlib import Path
import tempfile
import unittest
from src.pipeline.kb_builder import build_kb_v3
from scripts.audit_kb_candidate import audit


class CandidateAuditTests(unittest.TestCase):
    def test_valid_text_and_tampered_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("银行业务操作说明。" * 30, encoding="utf-8")
            subset = root / "subset.json"
            subset.write_text(json.dumps({"files": [{"rel_path": "a.txt", "ext": ".txt"}]}))
            build_kb_v3(root / "parsed", root / "kb", docs_root=root, subset_path=subset)
            result = audit(root / "kb")
            self.assertTrue(result["structural_integrity_passed"])
            self.assertEqual(result["physical_pages"], 0)
            chunks = root / "kb/chunks.jsonl"
            rows = [json.loads(line) for line in chunks.read_text(encoding="utf-8").splitlines()]
            rows[0]["block_ids"] = ["nonexistent"]
            chunks.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertFalse(audit(root / "kb")["structural_integrity_passed"])
