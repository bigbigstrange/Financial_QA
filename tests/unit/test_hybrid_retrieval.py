import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from src.system.rerank import EvidenceReranker
from src.system.retrieval import HybridRetriever
from src.system.sparse import SparseRetriever, tokenize_text
from src.system.structured import StructuredRetriever


@dataclass(frozen=True)
class StaticVisibility:
    versions: frozenset[str]

    def allowed(self, version_id: str) -> bool:
        return version_id in self.versions


class DenseFake:
    def search(self, query, top_k=10, visibility=None):
        return [
            {
                "block_id": "b1",
                "content": "存款金额",
                "document_version": "dv-active",
                "source_doc": "邮储规则.pdf",
                "score": 0.8,
                "type": "paragraph",
                "retrieval_channel": "text",
            }
        ]


class RemoteGraphFake:
    requires_remote_model = True

    def __init__(self):
        self.calls = 0

    def search(self, query, top_k=10, visibility=None):
        self.calls += 1
        return []


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = [
            {
                "block_id": "b1",
                "content": "邮储银行借记卡存款金额",
                "document_version": "dv-active",
                "source_doc": "邮政储蓄规则.pdf",
                "section_path": "存款",
                "type": "paragraph",
            },
            {
                "block_id": "b2",
                "content": "信用卡客服电话",
                "document_version": "dv-active",
                "source_doc": "邮储信用卡.pdf",
                "section_path": "信用卡",
                "type": "paragraph",
            },
            {
                "block_id": "b3",
                "content": "旧内容",
                "document_version": "dv-deleted",
                "source_doc": "已删除规则.pdf",
                "section_path": "旧文档",
                "type": "paragraph",
            },
        ]

    def test_tokenize_is_deterministic(self):
        self.assertEqual(tokenize_text("邮储银行"), tokenize_text("邮储银行"))
        self.assertTrue(tokenize_text("邮储银行"))

    def test_sparse_retriever_scores_and_marks_channel(self):
        retriever = SparseRetriever(self.blocks)
        results = retriever.search("存款 金额", top_k=5)

        self.assertTrue(results)
        self.assertGreater(results[0]["sparse_score"], 0)
        self.assertEqual(results[0]["retrieval_channel"], "sparse")
        self.assertIn("b1", [item["block_id"] for item in results])

    def test_sparse_postings_match_full_scan_scores(self):
        retriever = SparseRetriever(self.blocks)
        query = "邮储银行 存款 金额"
        tokens = tokenize_text(query)
        expected = sorted(
            ((idx, round(retriever._score(tokens, idx), 6))
             for idx in range(len(self.blocks))),
            key=lambda pair: (-pair[1], pair[0]),
        )
        expected = [(self.blocks[idx]["block_id"], score)
                    for idx, score in expected if score > 0]
        actual = [(item["block_id"], item["sparse_score"])
                  for item in retriever.search(query, top_k=10)]
        self.assertEqual(actual, expected)

    def test_structured_retriever_filters_by_institution(self):
        retriever = StructuredRetriever(self.blocks)
        results = retriever.search("邮储银行 存款", top_k=5)

        self.assertTrue(results)
        self.assertEqual(results[0]["block_id"], "b1")
        self.assertGreater(results[0]["structured_score"], 0)
        self.assertEqual(results[0]["retrieval_channel"], "structured")

    def test_hybrid_merges_channels_and_keeps_scores(self):
        sparse = SparseRetriever(self.blocks)
        structured = StructuredRetriever(self.blocks)
        hybrid = HybridRetriever(
            fusion_retriever=DenseFake(),
            sparse_retriever=sparse,
            structured_retriever=structured,
        )

        results = hybrid.search("邮储 存款", top_k=10)

        self.assertTrue(results)
        top = results[0]
        self.assertIn("dense_score", top)
        self.assertIn("sparse_score", top)
        self.assertIn("structured_score", top)
        self.assertIn("fusion_score", top)
        self.assertIn("fusion_rank", top)
        self.assertIn("retrieval_channels", top)
        self.assertGreaterEqual(top["fusion_rank"], 1)

    def test_local_only_query_skips_remote_graph_channel(self):
        remote = RemoteGraphFake()
        hybrid = HybridRetriever(fusion_retriever=DenseFake(), lightrag_retriever=remote)
        self.assertTrue(hybrid.search("存款", allow_remote_graph=False))
        self.assertEqual(remote.calls, 0)
        hybrid.search("存款")
        self.assertEqual(remote.calls, 1)

    def test_hybrid_filters_deleted_version(self):
        sparse = SparseRetriever(self.blocks)
        structured = StructuredRetriever(self.blocks)
        hybrid = HybridRetriever(
            fusion_retriever=DenseFake(),
            sparse_retriever=sparse,
            structured_retriever=structured,
        )
        visibility = StaticVisibility(frozenset({"dv-active"}))

        results = hybrid.search("旧内容 规则", top_k=10, visibility=visibility)

        self.assertTrue(
            all(item["document_version"] == "dv-active" for item in results)
        )
        self.assertNotIn("dv-deleted", [item["document_version"] for item in results])

    def test_heuristic_rerank_does_not_overwrite_fusion_score(self):
        with patch.dict(os.environ, {"FINANCE_RERANK_BACKEND": "heuristic"}):
            reranker = EvidenceReranker()
        item = {
            "block_id": "b1",
            "content": "邮储银行借记卡存款金额",
            "source_doc": "邮政储蓄规则.pdf",
            "fusion_score": 0.31,
            "score": 0.2,
            "type": "paragraph",
        }

        ranked = reranker.rerank("邮储 存款", [item], top_k=1)

        self.assertEqual(ranked[0]["fusion_score"], 0.31)
        self.assertIn("rerank_score", ranked[0])

    def test_default_rerank_without_api_preserves_rrf_order(self):
        with patch.dict(os.environ, {"FINANCE_RERANK_BACKEND": "hybrid",
                                  "FINANCE_DISABLE_PAID_API": "1"}):
            reranker = EvidenceReranker()
        evidence = [
            {"block_id": "right", "fusion_score": 0.04,
             "source_doc": "上海银行产品.pdf", "content": "简短规则"},
            {"block_id": "distractor", "fusion_score": 0.01,
             "source_doc": "上海银行简介.pdf", "content": "上海银行理财产品收益风险"},
        ]
        ranked = reranker.rerank("上海银行理财产品收益风险", evidence, top_k=2)
        self.assertEqual([item["block_id"] for item in ranked], ["right", "distractor"])


if __name__ == "__main__":
    unittest.main()
