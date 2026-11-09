import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import chat, runs
from app.core.deps import get_current_user
from app.db.base import Base
from app.db.session import get_db
from app.models import Message, User
from src.contracts import Evidence, ModelSnapshot, RunResult, RunStatus


class FakeRunService:
    def __init__(self):
        self.calls = 0
        self.by_key = {}
        self.by_id = {}

    def create_and_execute(self, db, *, user_id, request):
        self.calls += 1
        key = (user_id, request.idempotency_key)
        if request.idempotency_key and key in self.by_key:
            return self.by_key[key]
        run_id = f"run-{len(self.by_id) + 1}"
        result = RunResult(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            answer="统一答案",
            evidence=[
                Evidence(
                    evidence_id="ev-1",
                    document_id="doc-1",
                    document_version="v1",
                    block_id="b1",
                    text="依据",
                    source_name="规则.pdf",
                    page_no=2,
                )
            ],
            confidence=0.8,
            thinking="内部过程",
            model_snapshot=ModelSnapshot(
                snapshot_id="sha256:api",
                profile_name="stable-v1",
                provider="fake",
                requested_model="fake-1",
                actual_model="fake-1",
            ),
            latency_sec=0.01,
            memory={"rss_mb": 100},
        )
        record = SimpleNamespace(user_id=user_id, result=result)
        self.by_id[run_id] = record
        if request.idempotency_key:
            self.by_key[key] = result
        return result

    def get_owned(self, db, *, run_id, user_id):
        record = self.by_id.get(run_id)
        if record is None or record.user_id != user_id:
            return None
        return record

    def to_result(self, record, *, include_thinking=False):
        result = record.result
        if include_thinking:
            return result
        return result.model_copy(update={"thinking": None})


class RunsAPITests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            self.user = User(
                username="tester",
                email="tester@example.com",
                password_hash="unused",
                role="user",
            )
            db.add(self.user)
            db.commit()
            db.refresh(self.user)
            self.user_id = self.user.id

        self.current_user_id = self.user_id
        self.service = FakeRunService()
        self.original_runs_service = getattr(runs, "run_service", None)
        self.original_chat_service = getattr(chat, "run_service", None)
        runs.run_service = self.service
        chat.run_service = self.service

        app = FastAPI()
        app.include_router(runs.router, prefix="/api")
        app.include_router(chat.router, prefix="/api")

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        def override_user():
            return SimpleNamespace(id=self.current_user_id, role="user")

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_current_user] = override_user
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self.engine.dispose)

    def tearDown(self):
        if self.original_runs_service is not None:
            runs.run_service = self.original_runs_service
        if self.original_chat_service is not None:
            chat.run_service = self.original_chat_service

    def test_create_and_get_run(self):
        created = self.client.post(
            "/api/runs",
            json={"question": "问题", "idempotency_key": "request-api-1"},
        )
        self.assertEqual(created.status_code, 200)
        payload = created.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["evidence"][0]["block_id"], "b1")
        self.assertEqual(payload["model_snapshot"]["profile_name"], "stable-v1")

        fetched = self.client.get(f"/api/runs/{payload['run_id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["run_id"], payload["run_id"])

    def test_blank_question_is_rejected_before_service(self):
        response = self.client.post("/api/runs", json={"question": ""})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.service.calls, 0)

    def test_other_users_run_is_hidden_as_not_found(self):
        created = self.client.post("/api/runs", json={"question": "问题"}).json()
        self.current_user_id = self.user_id + 1

        response = self.client.get(f"/api/runs/{created['run_id']}")

        self.assertEqual(response.status_code, 404)

    def test_idempotency_key_returns_same_run(self):
        body = {"question": "问题", "idempotency_key": "request-api-2"}

        first = self.client.post("/api/runs", json=body).json()
        second = self.client.post("/api/runs", json=body).json()

        self.assertEqual(first["run_id"], second["run_id"])

    def test_legacy_chat_uses_run_service_and_keeps_response_shape(self):
        response = self.client.post(
            "/api/chat",
            json={"question": "问题", "show_thinking": False},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        for key in (
            "conversation_id",
            "user_message",
            "assistant_message",
            "elapsed_sec",
            "memory",
            "profile",
            "run_id",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["assistant_message"]["content"], "统一答案")
        self.assertIsNone(payload["assistant_message"]["thinking"])
        self.assertEqual(self.service.calls, 1)
        with self.Session() as db:
            messages = db.query(Message).all()
            self.assertEqual([item.role for item in messages], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
