import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import RunRecord
from app.schemas import RunCreateRequest
from app.services.run_service import RunService
from src.contracts import (
    KnowledgeSnapshotRef,
    ModelSnapshot,
    evidence_from_v2,
)


class FakeModelRegistry:
    def __init__(self, calls):
        self.calls = calls

    def snapshot(self):
        self.calls.append("model_snapshot")
        return ModelSnapshot(
            snapshot_id="model-a",
            profile_name="stable-v1",
            provider="fake",
            requested_model="fake-1",
            actual_model="fake-1",
        )


class FakeKnowledgeRegistry:
    def __init__(self, calls):
        self.calls = calls

    def active_ref(self, _db):
        self.calls.append("knowledge_snapshot")
        return KnowledgeSnapshotRef("snap-a", "snapshot.json")


class RecordingQA:
    def __init__(self, session, calls):
        self.session = session
        self.calls = calls

    def answer(self, question, question_id, knowledge_snapshot=None):
        record = self.session.get(RunRecord, question_id)
        assert record.knowledge_snapshot_id == "snap-a"
        self.calls.append("answer:snap-a")
        return {
            "answer": "答案",
            "evidence": [
                {
                    "document_id": "doc-a",
                    "document_version": "dv-a",
                    "block_id": "blk-a",
                    "content": "依据",
                    "source_name": "rules.pdf",
                    "page": 1,
                    "structural_locator": "page/000001/text/000001",
                }
            ],
            "confidence": 1.0,
        }


class RunKnowledgeSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()

    def test_knowledge_snapshot_is_committed_before_inference(self):
        calls = []
        service = RunService(
            qa_runtime=RecordingQA(self.session, calls),
            registry=FakeModelRegistry(calls),
            knowledge_registry=FakeKnowledgeRegistry(calls),
        )

        result = service.create_and_execute(
            self.session,
            user_id=1,
            request=RunCreateRequest(question="问题"),
        )

        self.assertEqual(
            calls,
            ["model_snapshot", "knowledge_snapshot", "answer:snap-a"],
        )
        self.assertEqual(result.knowledge_snapshot_id, "snap-a")
        self.assertEqual(result.evidence[0].evidence_id.startswith("ev_"), True)

    def test_evidence_id_includes_snapshot_and_version(self):
        raw = {
            "document_id": "doc-a",
            "document_version": "dv-a",
            "block_id": "blk-a",
            "content": "依据",
            "source_name": "rules.pdf",
            "page": 1,
            "structural_locator": "page/000001/text/000001",
        }

        one = evidence_from_v2(raw, "snap-a")
        two = evidence_from_v2(raw, "snap-b")

        self.assertNotEqual(one.evidence_id, two.evidence_id)
        self.assertEqual(one.document_version, "dv-a")
        self.assertEqual(one.metadata["structural_locator"], raw["structural_locator"])


if __name__ == "__main__":
    unittest.main()
