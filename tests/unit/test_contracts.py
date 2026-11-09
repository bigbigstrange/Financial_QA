import unittest

from pydantic import ValidationError

from src.contracts import (
    Evidence,
    ModelSnapshot,
    RunResult,
    RunStatus,
    evidence_from_legacy,
    evidence_from_v2,
)
from src.pipeline.docir import DocBlock, DocIR, DocPage


class ContractTests(unittest.TestCase):
    def test_web_evidence_keeps_public_source_url(self):
        item = evidence_from_legacy({"source_doc": "公告", "source_url": "https://example.com/a",
                                     "content": "利率 3%", "retrieval_channels": ["web"]}, 0)
        self.assertEqual(item.metadata["source_url"], "https://example.com/a")

    def test_cross_page_chunk_keeps_page_range(self):
        item = evidence_from_v2({"document_version": "dv1", "block_id": "ck1",
                                 "page": 1, "page_end": 2, "content": "跨页事实"}, "snap1")
        self.assertEqual((item.page_no, item.page_end), (1, 2))

    def test_evidence_keeps_score_types_separate(self):
        item = Evidence(
            evidence_id="ev-1",
            document_id="doc-1",
            document_version="v1",
            block_id="b1",
            text="监管要求",
            page_no=3,
            dense_score=0.8,
            rerank_score=2.4,
            retrieval_channels=["dense", "rerank"],
        )

        self.assertEqual(item.dense_score, 0.8)
        self.assertEqual(item.rerank_score, 2.4)
        self.assertNotIn("confidence", item.model_dump())

    def test_model_snapshot_is_immutable(self):
        snapshot = ModelSnapshot(
            snapshot_id="sha256:abc",
            profile_name="stable-v1",
            provider="deepseek",
            requested_model="deepseek-chat",
            actual_model="deepseek-chat",
        )

        with self.assertRaises(ValidationError):
            snapshot.actual_model = "changed"

    def test_model_snapshot_nested_configuration_is_immutable(self):
        snapshot = ModelSnapshot(
            snapshot_id="sha256:nested",
            profile_name="stable-v1",
            provider="deepseek",
            generation_parameters={"temperature": 0.2},
            retrieval_config={"channels": ["dense", "sparse"]},
        )

        with self.assertRaises(TypeError):
            snapshot.generation_parameters["temperature"] = 0.9
        with self.assertRaises(TypeError):
            snapshot.retrieval_config["channels"][0] = "changed"
        self.assertIn('"temperature":0.2', snapshot.model_dump_json())

    def test_run_result_rejects_completed_without_answer(self):
        with self.assertRaises(ValidationError):
            RunResult(run_id="r1", status=RunStatus.COMPLETED, answer=None)

    def test_existing_docir_constructor_still_serializes(self):
        doc = DocIR(doc_id="old", source_path="a.pdf", source_format="pdf")
        doc.add_page(
            DocPage(
                page_no=1,
                blocks=[DocBlock("b1", "text", "内容", 1)],
            )
        )

        data = doc.to_dict()

        self.assertEqual(data["document_id"], "old")
        self.assertTrue(data["document_version"])
        self.assertEqual(data["pages"][0]["blocks"][0]["block_id"], "b1")

    def test_legacy_evidence_missing_location_is_explicit(self):
        item = evidence_from_legacy(
            {"text": "内容", "source_doc": "a.pdf"},
            position=0,
        )

        self.assertTrue(item.evidence_id)
        self.assertIsNone(item.page_no)
        self.assertIn("missing_block_id", item.quality_flags)
        self.assertIn("missing_location", item.quality_flags)

    def test_legacy_evidence_id_is_deterministic_and_preserves_raw_score(self):
        raw = {"text": "内容", "source_doc": "a.pdf", "score": 0.42}

        first = evidence_from_legacy(raw, position=2)
        second = evidence_from_legacy(raw, position=2)

        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(first.metadata["legacy_score"], 0.42)
        self.assertIsNone(first.dense_score)


if __name__ == "__main__":
    unittest.main()
