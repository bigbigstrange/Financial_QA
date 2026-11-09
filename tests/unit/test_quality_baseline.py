from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import quality_baseline


class QualityBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.workbook = root / "questions.xlsx"
        self.workbook.write_bytes(b"fixture")
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps({"snapshot_id": "snap_test"}), encoding="utf-8")
        self.chunks = root / "chunks.jsonl"
        self.chunks.write_text(json.dumps({
            "chunk_id": "ck_1", "snapshot_id": "snap_test",
            "document_version": "dv_1", "content": "原文",
        }) + "\n", encoding="utf-8")
        self.review = root / "review.jsonl"
        self.source = [{"case_id": "finance500-r0002", "sheet_row": 2,
                        "question_id": "1", "question": "某产品是什么？"}]

    def _prepare(self):
        with patch.object(quality_baseline, "questions_from_workbook", return_value=self.source):
            quality_baseline.prepare(self.workbook, self.manifest, self.review)

    def _validate(self):
        with patch.object(quality_baseline, "questions_from_workbook", return_value=self.source):
            return quality_baseline.validate(
                self.workbook, self.manifest, self.chunks, self.review)

    def test_pending_case_never_passes_and_review_is_not_overwritten(self):
        self._prepare()
        report = self._validate()
        self.assertFalse(report["quality_gate_ready"])
        self.assertEqual(report["pending_cases"], 1)
        with self.assertRaises(FileExistsError):
            self._prepare()

    def test_approved_requires_human_and_current_source_evidence(self):
        self._prepare()
        row = json.loads(self.review.read_text(encoding="utf-8"))
        row.update(review_status="approved", reviewer="人工审核人",
                   expected_answer="参考事实", expected_refusal=False,
                   gold_evidence_ids=["ck_missing"])
        self.review.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        self.assertFalse(self._validate()["quality_gate_ready"])
        row["gold_evidence_ids"] = ["ck_1"]
        self.review.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        self.assertTrue(self._validate()["quality_gate_ready"])

    def test_snapshot_mismatch_blocks_review(self):
        self._prepare()
        row = json.loads(self.review.read_text(encoding="utf-8"))
        row.update(review_status="refuse", reviewer="人工审核人",
                   expected_refusal=True, source_snapshot="old")
        self.review.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        report = self._validate()
        self.assertFalse(report["quality_gate_ready"])
        self.assertTrue(any("快照不匹配" in error for error in report["errors"]))
