import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import Conversation, RunRecord
from app.schemas import RunCreateRequest
from app.services.run_service import RunService
from src.contracts import ErrorCode, ModelSnapshot, RunStatus
from src.contracts import ContractError


class FakeRegistry:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def snapshot(self):
        self.calls.append("snapshot")
        return ModelSnapshot(
            snapshot_id="sha256:run-test",
            profile_name="stable-v1",
            provider="fake",
            requested_model="fake-1",
            actual_model="fake-1",
        )


class FakeQA:
    def __init__(self, result=None, error=None, calls=None):
        self.result = result or {
            "answer": "旧答案",
            "evidence": [
                {
                    "text": "依据",
                    "source_doc": "规则.pdf",
                    "block_id": "b1",
                    "page": 2,
                    "score": 0.8,
                }
            ],
            "confidence": 0.7,
            "thinking": "内部",
            "_memory": {"rss_mb": 100},
            "_profile": "stable-v1",
        }
        self.error = error
        self.call_count = 0
        self.calls = calls if calls is not None else []

    def answer(self, question, question_id, knowledge_snapshot=None):
        self.calls.append("answer")
        self.call_count += 1
        _ = knowledge_snapshot
        if self.error:
            raise self.error
        return dict(self.result)


class RunServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db = self.Session()
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.db.close)

    def service(self, *, qa=None, registry=None):
        return RunService(
            qa_runtime=qa or FakeQA(),
            registry=registry or FakeRegistry(),
        )

    def test_completed_run_is_persisted_with_snapshot_and_evidence(self):
        service = self.service()
        request = RunCreateRequest(
            question="问题",
            idempotency_key="request-0001",
        )

        # This fake-runtime contract must not depend on the caller's paid-API env.
        with patch.dict("os.environ", {"FINANCE_DISABLE_PAID_API": "0"}):
            result = service.create_and_execute(self.db, user_id=1, request=request)

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.answer, "旧答案")
        self.assertEqual(result.evidence[0].block_id, "b1")
        self.assertEqual(result.model_snapshot.snapshot_id, "sha256:run-test")
        record = self.db.get(RunRecord, result.run_id)
        self.assertEqual(record.status, RunStatus.COMPLETED.value)
        self.assertEqual(service._load_options(record),
                         {"use_paid_model": True, "use_web_search": True})

    def test_failed_ark_request_records_requested_provider_not_old_deepseek(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"FINANCE_LLM_PROVIDER": "ark",
                                  "FINANCE_DISABLE_PAID_API": "0",
                                  "ARK_MODEL": "doubao-seed-2-1-pro-260915"}):
            result = self.service(qa=FakeQA(error=RuntimeError("offline"))).create_and_execute(
                self.db, user_id=1, request=RunCreateRequest(question="问题"))
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.model_snapshot.provider, "ark")
        self.assertEqual(result.model_snapshot.requested_model, "doubao-seed-2-1-pro-260915")
        self.assertIsNone(result.model_snapshot.actual_model)

    def test_same_user_idempotency_key_does_not_execute_twice(self):
        qa = FakeQA()
        service = self.service(qa=qa)
        request = RunCreateRequest(
            question="问题",
            idempotency_key="request-0002",
        )

        first = service.create_and_execute(self.db, user_id=1, request=request)
        second = service.create_and_execute(self.db, user_id=1, request=request)

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(qa.call_count, 1)

    def test_idempotency_key_cannot_change_service_switches(self):
        service = self.service()
        key = "request-options"
        first = service.create_and_execute(self.db, user_id=1,
            request=RunCreateRequest(question="问题", idempotency_key=key,
                                     use_paid_model=False, use_web_search=False))
        self.assertEqual(first.model_snapshot.provider, "local")
        self.assertEqual(service._load_options(self.db.get(RunRecord, first.run_id)),
                         {"use_paid_model": False, "use_web_search": False})
        with self.assertRaises(ContractError) as raised:
            service.create_and_execute(self.db, user_id=1,
                request=RunCreateRequest(question="问题", idempotency_key=key,
                                         use_paid_model=True, use_web_search=False))
        self.assertEqual(raised.exception.code, ErrorCode.RUN_CONFLICT)

    def test_reusing_idempotency_key_for_different_question_conflicts(self):
        service = self.service()
        service.create_and_execute(
            self.db,
            user_id=1,
            request=RunCreateRequest(
                question="第一个问题",
                idempotency_key="request-conflict",
            ),
        )

        with self.assertRaises(ContractError) as raised:
            service.create_and_execute(
                self.db,
                user_id=1,
                request=RunCreateRequest(
                    question="另一个问题",
                    idempotency_key="request-conflict",
                ),
            )

        self.assertEqual(raised.exception.code, ErrorCode.RUN_CONFLICT)

    def test_concurrent_idempotency_race_returns_existing_run(self):
        qa = FakeQA()
        service = self.service(qa=qa)
        request = RunCreateRequest(
            question="问题",
            idempotency_key="request-race",
        )
        existing = service.create_and_execute(self.db, user_id=1, request=request)
        original_query = self.db.query
        hidden = False

        class HiddenQuery:
            def filter(self, *args, **kwargs):
                return self

            def first(self):
                return None

        def hide_first_run_query(*entities, **kwargs):
            nonlocal hidden
            if not hidden and entities == (RunRecord,):
                hidden = True
                return HiddenQuery()
            return original_query(*entities, **kwargs)

        self.db.query = hide_first_run_query

        raced = service.create_and_execute(self.db, user_id=1, request=request)

        self.assertEqual(raced.run_id, existing.run_id)
        self.assertEqual(qa.call_count, 1)

    def test_idempotency_key_is_scoped_to_user(self):
        qa = FakeQA()
        service = self.service(qa=qa)
        request = RunCreateRequest(
            question="问题",
            idempotency_key="request-0003",
        )

        one = service.create_and_execute(self.db, user_id=1, request=request)
        two = service.create_and_execute(self.db, user_id=2, request=request)

        self.assertNotEqual(one.run_id, two.run_id)
        self.assertEqual(qa.call_count, 2)

    def test_snapshot_is_frozen_before_qa_execution(self):
        calls = []
        service = self.service(
            qa=FakeQA(calls=calls),
            registry=FakeRegistry(calls=calls),
        )

        service.create_and_execute(
            self.db,
            user_id=1,
            request=RunCreateRequest(question="问题"),
        )

        self.assertEqual(calls, ["snapshot", "answer"])

    def test_conversation_must_belong_to_run_owner(self):
        conversation = Conversation(user_id=2, title="other")
        self.db.add(conversation)
        self.db.commit()

        with self.assertRaises(ContractError) as raised:
            self.service().create_and_execute(
                self.db,
                user_id=1,
                request=RunCreateRequest(
                    question="问题",
                    conversation_id=conversation.id,
                ),
            )

        self.assertEqual(raised.exception.code, ErrorCode.RUN_NOT_FOUND)

    def test_thinking_is_only_persisted_when_requested(self):
        service = self.service()
        hidden = service.create_and_execute(
            self.db,
            user_id=1,
            request=RunCreateRequest(
                question="隐藏",
                idempotency_key="thinking-hidden",
                show_thinking=False,
            ),
        )
        shown = service.create_and_execute(
            self.db,
            user_id=1,
            request=RunCreateRequest(
                question="显示",
                idempotency_key="thinking-shown",
                show_thinking=True,
            ),
        )

        self.assertIsNone(hidden.thinking)
        self.assertIsNone(self.db.get(RunRecord, hidden.run_id).thinking)
        self.assertEqual(shown.thinking, "内部")
        self.assertEqual(self.db.get(RunRecord, shown.run_id).thinking, "内部")

    def test_runtime_failures_are_persisted_with_stable_codes(self):
        cases = [
            (MemoryError("limit"), ErrorCode.MEMORY_LIMIT),
            (RuntimeError("busy"), ErrorCode.INFERENCE_BUSY),
            (ValueError("secret detail"), ErrorCode.INTERNAL_ERROR),
        ]
        for index, (error, code) in enumerate(cases):
            with self.subTest(code=code):
                result = self.service(qa=FakeQA(error=error)).create_and_execute(
                    self.db,
                    user_id=1,
                    request=RunCreateRequest(
                        question="问题",
                        idempotency_key=f"request-error-{index}",
                    ),
                )
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertEqual(result.error_code, code)
                if code == ErrorCode.INTERNAL_ERROR:
                    self.assertNotIn("secret detail", result.error_message)

    def test_final_commit_failure_never_returns_completed(self):
        service = self.service()
        commit_count = 0

        def fail_third_commit(_session):
            nonlocal commit_count
            commit_count += 1
            if commit_count == 3:
                raise RuntimeError("database unavailable")

        event.listen(self.db, "before_commit", fail_third_commit)
        self.addCleanup(event.remove, self.db, "before_commit", fail_third_commit)

        result = service.create_and_execute(
            self.db,
            user_id=1,
            request=RunCreateRequest(question="问题"),
        )

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.error_code, ErrorCode.INTERNAL_ERROR)
        self.assertEqual(self.db.get(RunRecord, result.run_id).status, "failed")


if __name__ == "__main__":
    unittest.main()
