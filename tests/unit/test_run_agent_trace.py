"""Agent 轨迹的持久化与契约测试。

验证「跑一次 graph 模式的问答 → 轨迹落库 → 能按 run_id 读回」这条链路，
以及 legacy 模式下不写轨迹、字段保持默认（保证可回退）。
"""
import json
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models import RunRecord
from app.schemas import RunCreateRequest
from app.services.run_service import RunService
from src.contracts import ModelSnapshot, RunStatus

TRACE_PAYLOAD = {
    "question_id": "q-1",
    "question": "某银行2023年营业收入是多少？",
    "answer": "2023 年营业收入 1,000 亿元，同比增长 5%[1]。",
    "knowledge_points": [
        {
            "chunk_id": "ck_1",
            "block_id": "ck_1",
            "document_id": "doc_1",
            "document_version": "dv_1",
            "content": "2023年实现营业收入1000亿元，同比增长5%。",
            "source_doc": "某银行2023年报.pdf",
            "page": 3,
            "section_path": "经营概况",
        }
    ],
    "evidence": [
        {
            "chunk_id": "ck_1",
            "block_id": "ck_1",
            "document_id": "doc_1",
            "document_version": "dv_1",
            "content": "2023年实现营业收入1000亿元，同比增长5%。",
            "source_doc": "某银行2023年报.pdf",
            "page": 3,
            "section_path": "经营概况",
        }
    ],
    "confidence": 0.83,
    "agent_mode": "graph",
    "route": "retrieval",
    "retry_retrieve": 1,
    "retry_generate": 0,
    "trace": [
        {"node": "route_query", "elapsed_ms": 12, "decision": "retrieval",
         "detail": {"reason": "需要查业务规则"}},
        {"node": "retrieve", "elapsed_ms": 820, "decision": "8 条证据"},
        {"node": "grade_documents", "elapsed_ms": 1100, "decision": "保留 8/12"},
        {"node": "generate", "elapsed_ms": 2400, "decision": "6 字"},
        {"node": "check_hallucination", "elapsed_ms": 600, "decision": "True"},
        {"node": "check_answer", "elapsed_ms": 580, "decision": "True"},
    ],
    "_memory": {"rss_mb": 1234.0},
}


class FakeRegistry:
    def snapshot(self) -> ModelSnapshot:
        return ModelSnapshot(
            snapshot_id="runtime:test",
            profile_name="test",
            provider="fake",
            requested_model="fake",
            actual_model="fake",
        )


class FakeRuntime:
    def __init__(self, payload: dict):
        self.payload = payload

    def answer(self, question, question_id, knowledge_snapshot=None):
        return dict(self.payload)


def _session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


class AgentTracePersistenceTests(unittest.TestCase):
    def _run(self, payload: dict) -> tuple[RunService, RunRecord]:
        db = _session()
        service = RunService(qa_runtime=FakeRuntime(payload), registry=FakeRegistry())
        result = service.create_and_execute(
            db, user_id=1, request=RunCreateRequest(question=payload["question"])
        )
        record = db.get(RunRecord, result.run_id)
        return service, record

    def test_graph_trace_is_persisted_and_read_back(self):
        service, record = self._run(TRACE_PAYLOAD)
        self.assertIsNotNone(record.agent_trace_json)
        stored = json.loads(record.agent_trace_json)
        self.assertEqual(stored["agent_mode"], "graph")
        self.assertEqual(stored["route"], "retrieval")
        self.assertEqual(stored["retry_retrieve"], 1)
        self.assertEqual(len(stored["trace"]), 6)

        result = service.to_result(record)
        self.assertEqual(result.agent_mode, "graph")
        self.assertEqual(result.route, "retrieval")
        self.assertEqual(result.retry_retrieve, 1)
        self.assertEqual(result.retry_generate, 0)
        self.assertEqual([t["node"] for t in result.trace][0], "route_query")

    def test_actual_runtime_model_replaces_configured_legacy_snapshot(self):
        payload = {**TRACE_PAYLOAD, "_model_snapshot": ModelSnapshot(
            snapshot_id="runtime:ark", profile_name="runtime", provider="ark",
            requested_model="doubao-seed-2-1-pro-260915",
            actual_model="doubao-seed-2-1-pro-260915",
        )}
        service, record = self._run(payload)
        assert service.to_result(record).model_snapshot.provider == "ark"
        assert service.to_result(record).model_snapshot.actual_model == "doubao-seed-2-1-pro-260915"

    def test_local_only_run_does_not_claim_deepseek_or_ark(self):
        service, record = self._run({**TRACE_PAYLOAD, "_local_only": True})
        snapshot = service.to_result(record).model_snapshot
        assert snapshot.provider == "local"
        assert snapshot.actual_model is None

    def test_trace_reports_verification_truth_to_normal_chat(self):
        service, record = self._run({**TRACE_PAYLOAD, "verified": True})
        assert service.to_trace(record)["verified"] is True

    def test_to_trace_endpoint_payload_shape(self):
        service, record = self._run(TRACE_PAYLOAD)
        payload = service.to_trace(record)
        self.assertEqual(payload["run_id"], record.id)
        self.assertEqual(payload["agent_mode"], "graph")
        self.assertEqual(len(payload["trace"]), 6)
        self.assertNotIn("note", payload)
        for entry in payload["trace"]:
            self.assertIn("node", entry)
            self.assertIn("elapsed_ms", entry)

    def test_legacy_mode_writes_no_trace_and_keeps_defaults(self):
        legacy_payload = {
            "question_id": "q-2",
            "question": "某银行2023年营业收入是多少？",
            "answer": "legacy 答案",
            "knowledge_points": [],
            "evidence": [],
            "confidence": 0.5,
        }
        service, record = self._run(legacy_payload)
        self.assertIsNone(record.agent_trace_json)

        result = service.to_result(record)
        self.assertIsNone(result.agent_mode)
        self.assertIsNone(result.route)
        self.assertEqual(result.retry_retrieve, 0)
        self.assertEqual(result.retry_generate, 0)
        self.assertEqual(result.trace, [])

        payload = service.to_trace(record)
        self.assertEqual(payload["trace"], [])
        self.assertIn("note", payload)

    def test_failed_run_still_returns_contract_defaults(self):
        class ExplodingRuntime:
            def answer(self, *a, **kw):
                raise RuntimeError("模拟推理失败")

        db = _session()
        service = RunService(qa_runtime=ExplodingRuntime(), registry=FakeRegistry())
        result = service.create_and_execute(
            db, user_id=1, request=RunCreateRequest(question="会失败的问题")
        )
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIsNotNone(result.error_code)
        self.assertEqual(result.trace, [])

    def test_malformed_trace_json_does_not_crash(self):
        service, record = self._run(TRACE_PAYLOAD)
        record.agent_trace_json = "{ not json"
        result = service.to_result(record)
        self.assertEqual(result.trace, [])
        self.assertIsNone(result.agent_mode)


if __name__ == "__main__":
    unittest.main()
