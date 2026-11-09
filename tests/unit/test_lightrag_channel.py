"""LightRAG 图通道契约测试。

最关键的一条：**图通道失败绝不能影响其余三路检索**。
LightRAG 是独立进程/独立服务，它挂了、超时了、返回垃圾了，
问答都必须照常给出结果 —— 历史上 `_EmptyImageRetriever` 的签名不匹配
让整条融合检索静默返回 0 条，正是这里要防的同类故障。
"""
import unittest
import os
from unittest.mock import patch

from src.system.lightrag_channel import (
    LightRAGRetriever,
    build_lightrag_retriever,
)
from src.system.lightrag_client import LightRAGClient, LightRAGConfig, LightRAGError


def _blocks():
    return [
        {"chunk_id": "ck_1", "block_id": "ck_1", "content": "同业业务是指以金融同业为交易对手方。",
         "source_doc": "同业业务通知.pdf", "page_start": 1, "page": 1,
         "section_path": "总则", "document_version": "dv_1", "type": "paragraph"},
        {"chunk_id": "ck_2", "block_id": "ck_2", "content": "商业银行应按季度报送数据。",
         "source_doc": "同业业务通知.pdf", "page_start": 3, "page": 3,
         "section_path": "报送", "document_version": "dv_1", "type": "paragraph"},
    ]


class _FakeClient:
    """可控的假客户端：用于模拟正常/异常/垃圾响应。"""

    def __init__(self, payload=None, *, available=True, raise_on_query=None):
        self.payload = payload or {}
        self._available = available
        self._raise = raise_on_query
        self.config = LightRAGConfig(base_url="http://fake")
        self.calls = []

    def available(self) -> bool:
        return self._available

    def query_data(self, question, **kwargs):
        self.calls.append({"question": question, **kwargs})
        if self._raise is not None:
            raise self._raise
        return self.payload


def _retriever(client):
    return LightRAGRetriever(_blocks(), client=client, log=lambda _msg: None)


class SourceSplittingTests(unittest.TestCase):
    def test_splits_common_separators(self):
        r = _retriever(_FakeClient())
        for raw in ("ck_1,ck_2", "ck_1、ck_2", "ck_1|ck_2", "ck_1;ck_2",
                    "ck_1<SEP>ck_2", ["ck_1", "ck_2"], ("ck_1", "ck_2")):
            self.assertEqual(sorted(r._split_sources(raw)), ["ck_1", "ck_2"], raw)

    def test_handles_empty_and_none(self):
        r = _retriever(_FakeClient())
        self.assertEqual(r._split_sources(None), [])
        self.assertEqual(r._split_sources(""), [])
        self.assertEqual(r._split_sources("  ,  "), [])


class MappingTests(unittest.TestCase):
    def test_structured_response_aligns_content_without_guessing_first_page(self):
        blocks = _blocks()
        blocks[1]["content"] = "季度报送的业务数据应当包括金融同业交易对手名称、交易金额、到期日期及风险分类，并由商业银行审核。"
        payload = {"status": "success", "data": {"chunks": [{
            "chunk_id": "server-generated-id", "file_path": "同业业务通知.pdf",
            "content": blocks[1]["content"],
        }]}}
        r = LightRAGRetriever(blocks, client=_FakeClient(payload))
        self.assertEqual([b["chunk_id"] for b in r.search("季度报送")], ["ck_2"])

    def test_document_name_alone_cannot_assign_first_chunk(self):
        payload = {"entities": [{"name": "同业业务", "source_id": "同业业务通知.pdf"}]}
        self.assertEqual(_retriever(_FakeClient(payload)).search("同业业务"), [])

    def _payload(self):
        return {
            "entities": [
                {"entity_name": "同业业务", "source_id": "ck_1"},
                {"entity_name": "商业银行", "source_id": "ck_2"},
            ],
            "relations": [
                {"src_id": "同业业务", "tgt_id": "商业银行", "source_id": "ck_1,ck_2"},
            ],
            "chunks": [{"chunk_id": "ck_1"}],
        }

    def test_maps_entities_relations_and_chunks(self):
        out = _retriever(_FakeClient(self._payload())).search("同业业务", top_k=5)
        ids = {item["chunk_id"] for item in out}
        self.assertEqual(ids, {"ck_1", "ck_2"})
        for item in out:
            self.assertEqual(item["retrieval_channel"], "graph_lightrag")
            self.assertEqual(item["block_id"], item["chunk_id"])
            self.assertIn("graph_score", item)
            self.assertTrue(item["kg_mapped_exactly"])

    def test_weights_entity_above_relation_above_chunk(self):
        out = _retriever(_FakeClient(self._payload())).search("同业业务", top_k=5)
        by_id = {item["chunk_id"]: item for item in out}
        # ck_1 命中 1 实体 + 1 关系 + 1 原文块；ck_2 命中 1 实体 + 1 关系
        self.assertGreater(by_id["ck_1"]["graph_raw_score"],
                           by_id["ck_2"]["graph_raw_score"])
        self.assertEqual(by_id["ck_1"]["graph_score"], 1.0)

    def test_carries_kg_entities_and_relations(self):
        out = _retriever(_FakeClient(self._payload())).search("同业业务", top_k=5)
        by_id = {item["chunk_id"]: item for item in out}
        self.assertIn("同业业务", by_id["ck_1"]["kg_entities"])
        self.assertTrue(by_id["ck_1"]["kg_relations"])
        self.assertIn("同业业务→商业银行", by_id["ck_1"]["kg_relations"][0])

    def test_evidence_keeps_traceability_fields(self):
        out = _retriever(_FakeClient(self._payload())).search("同业业务", top_k=5)
        item = next(i for i in out if i["chunk_id"] == "ck_1")
        self.assertEqual(item["source_doc"], "同业业务通知.pdf")
        self.assertEqual(item["page_start"], 1)
        self.assertEqual(item["section_path"], "总则")
        self.assertEqual(item["document_version"], "dv_1")

    def test_falls_back_to_locator_when_chunk_id_unknown(self):
        """LightRAG 没保留我们的 chunk_id 时，按「文档名+页码」退化匹配。"""
        payload = {"entities": [{"entity_name": "同业业务",
                                 "source_id": "同业业务通知.pdf#p3"}]}
        out = _retriever(_FakeClient(payload)).search("同业业务", top_k=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["chunk_id"], "ck_2")     # 页码 3 对应 ck_2
        self.assertFalse(out[0]["kg_mapped_exactly"])

    def test_unmappable_sources_are_dropped(self):
        payload = {"entities": [{"entity_name": "X", "source_id": "完全不存在的来源"}]}
        out = _retriever(_FakeClient(payload)).search("X", top_k=5)
        self.assertEqual(out, [])

    def test_respects_top_k(self):
        out = _retriever(_FakeClient(self._payload())).search("同业业务", top_k=1)
        self.assertEqual(len(out), 1)


class DegradationTests(unittest.TestCase):
    """图通道的任何故障都必须静默降级为空列表。"""

    def test_unavailable_service_returns_empty(self):
        out = _retriever(_FakeClient(available=False)).search("问题", top_k=5)
        self.assertEqual(out, [])

    def test_query_error_returns_empty(self):
        client = _FakeClient(raise_on_query=LightRAGError("HTTP 500"))
        out = _retriever(client).search("问题", top_k=5)
        self.assertEqual(out, [])
        self.assertEqual(client.calls and 1 or 0, 1)

    def test_unexpected_exception_returns_empty(self):
        client = _FakeClient(raise_on_query=RuntimeError("连接被重置"))
        out = _retriever(client).search("问题", top_k=5)
        self.assertEqual(out, [])

    def test_malformed_payload_returns_empty(self):
        for payload in ({}, {"entities": "不是列表"}, {"entities": [None, 42]},
                        {"chunks": [{}]}):
            out = _retriever(_FakeClient(payload)).search("问题", top_k=5)
            self.assertEqual(out, [], f"payload={payload}")

    def test_stats_track_failures(self):
        client = _FakeClient(raise_on_query=LightRAGError("boom"))
        r = _retriever(client)
        r.search("问题", top_k=5)
        self.assertEqual(r.stats["queries"], 1)
        self.assertEqual(r.stats["errors"], 1)

    def test_availability_is_probed_once_and_resettable(self):
        client = _FakeClient(available=False)
        r = _retriever(client)
        r.search("问题", top_k=5)
        self.assertFalse(r.available)
        client._available = True
        self.assertFalse(r.available)      # 仍用缓存
        r.reset_availability()
        self.assertTrue(r.available)       # 重置后重新探测


class VisibilityTests(unittest.TestCase):
    def test_visibility_filter_is_applied(self):
        class _Vis:
            def allowed(self, document_version: str) -> bool:
                return document_version == "dv_other"

        payload = {"entities": [{"entity_name": "同业业务", "source_id": "ck_1"}]}
        out = _retriever(_FakeClient(payload)).search(
            "同业业务", top_k=5, visibility=_Vis()
        )
        self.assertEqual(out, [])


class BuildSwitchTests(unittest.TestCase):
    def setUp(self):
        # Tests of the graph switch are independent of the operator's cost pause.
        patcher = patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
    def test_disabled_by_default(self):
        self.assertIsNone(build_lightrag_retriever(_blocks(), enabled=False))

    def test_explicit_enable_builds_channel(self):
        r = build_lightrag_retriever(_blocks(), enabled=True,
                                     client=_FakeClient(), log=lambda _m: None)
        self.assertIsInstance(r, LightRAGRetriever)

    def test_env_switch(self):
        import os

        old = os.environ.get("FINANCE_ENABLE_GRAPH_CHANNEL")
        try:
            os.environ["FINANCE_ENABLE_GRAPH_CHANNEL"] = "1"
            self.assertIsNotNone(build_lightrag_retriever(
                _blocks(), client=_FakeClient(), log=lambda _m: None))
            os.environ["FINANCE_ENABLE_GRAPH_CHANNEL"] = "0"
            self.assertIsNone(build_lightrag_retriever(
                _blocks(), client=_FakeClient(), log=lambda _m: None))
        finally:
            if old is None:
                os.environ.pop("FINANCE_ENABLE_GRAPH_CHANNEL", None)
            else:
                os.environ["FINANCE_ENABLE_GRAPH_CHANNEL"] = old


class HybridIntegrationTests(unittest.TestCase):
    """第 4 通道并入 HybridRetriever 的 RRF 融合。"""

    def _hybrid(self, lightrag, graph_weight=1.0):
        from src.system.retrieval import (
            FusionRetriever,
            HybridRetriever,
            _EmptyImageRetriever,
        )
        from src.system.sparse import SparseRetriever
        from src.system.structured import StructuredRetriever

        class _TextRetriever:
            def search(self, query, top_k=10, visibility=None):
                out = []
                for b in _blocks()[:top_k]:
                    item = dict(b)
                    item["retrieval_channel"] = "text"
                    item["score"] = 0.9
                    out.append(item)
                return out

        docs = _blocks()
        return HybridRetriever(
            fusion_retriever=FusionRetriever(
                text_retriever=_TextRetriever(),
                image_retriever=_EmptyImageRetriever(),
            ),
            sparse_retriever=SparseRetriever(docs),
            structured_retriever=StructuredRetriever(docs),
            lightrag_retriever=lightrag,
            graph_weight=graph_weight,
        )

    def test_lightrag_rrf_contributes_to_fusion_score(self):
        payload = {"entities": [{"entity_name": "同业业务", "source_id": "ck_2"}]}
        lightrag = _retriever(_FakeClient(payload))
        hybrid = self._hybrid(lightrag)
        out = hybrid.search("同业业务是指什么", top_k=5)
        target = next(i for i in out if i["chunk_id"] == "ck_2")
        self.assertGreater(target["lightrag_rrf"], 0)
        self.assertIn("graph_lightrag", target["retrieval_channels"])
        self.assertGreater(target["fusion_score"], 0)

    def test_graph_weight_zero_disables_the_channel(self):
        payload = {"entities": [{"entity_name": "同业业务", "source_id": "ck_2"}]}
        hybrid = self._hybrid(_retriever(_FakeClient(payload)), graph_weight=0.0)
        out = hybrid.search("同业业务是指什么", top_k=5)
        for item in out:
            self.assertEqual(item.get("lightrag_rrf", 0.0), 0.0)

    def test_hybrid_works_when_lightrag_is_down(self):
        """关键回归：图服务挂掉时，其余三路仍必须返回证据。"""
        hybrid = self._hybrid(_retriever(_FakeClient(available=False)))
        out = hybrid.search("同业业务是指什么", top_k=5)
        self.assertTrue(out, "图通道不可用时混合检索返回了空结果")
        for item in out:
            self.assertEqual(item.get("lightrag_rrf", 0.0), 0.0)

    def test_hybrid_works_when_lightrag_raises(self):
        hybrid = self._hybrid(_retriever(_FakeClient(raise_on_query=RuntimeError("boom"))))
        out = hybrid.search("同业业务是指什么", top_k=5)
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
