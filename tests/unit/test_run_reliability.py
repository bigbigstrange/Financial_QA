import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models import RunEventRecord, RunRecord, ReviewTicketRecord
from app.schemas import RunCreateRequest
from app.services.run_service import RunService
from src.contracts import ModelSnapshot, RunStatus


class Registry:
    def snapshot(self):
        return ModelSnapshot(snapshot_id="test", profile_name="test", provider="local")


class Runtime:
    def answer(self, question, question_id, knowledge_snapshot=None, on_node=None, cancel_check=None):
        if on_node:
            on_node({"node": "retrieve", "decision": "found", "elapsed_ms": 2,
                     "detail": {"secret": "never-store"}})
        return {"answer": "answer", "evidence": [], "verified": False}


def _db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)()


def test_node_checkpoint_and_events_are_durable_and_redacted():
    engine, db = _db()
    try:
        service = RunService(qa_runtime=Runtime(), registry=Registry())
        result = service.create_and_execute(db, user_id=1,
            request=RunCreateRequest(question="private question"), on_node=lambda _: None)
        events = db.query(RunEventRecord).filter_by(run_id=result.run_id).order_by(RunEventRecord.id).all()
        assert [event.event for event in events] == ["started", "node", "done"]
        assert [event.id for event in events] == sorted(event.id for event in events)
        assert "never-store" not in "".join(event.data_json for event in events)
        assert "private question" not in "".join(event.data_json for event in events)
        checkpoint = json.loads(db.get(RunRecord, result.run_id).checkpoint_json)
        assert checkpoint["node"] == "retrieve"
        assert db.query(ReviewTicketRecord).filter_by(run_id=result.run_id).count() == 1
    finally:
        db.close()
        engine.dispose()


def test_failed_run_opens_one_review_ticket():
    class FailedRuntime:
        def answer(self, *_args, **_kwargs):
            raise ValueError("internal secret")

    engine, db = _db()
    try:
        service = RunService(qa_runtime=FailedRuntime(), registry=Registry())
        result = service.create_and_execute(db, user_id=1,
            request=RunCreateRequest(question="question"))
        assert result.status == RunStatus.FAILED
        assert db.query(ReviewTicketRecord).filter_by(run_id=result.run_id).count() == 1
        assert [event.event for event in db.query(RunEventRecord).filter_by(run_id=result.run_id)] == ["started", "error"]
    finally:
        db.close()
        engine.dispose()


def test_failure_log_omits_raw_exception_detail(caplog):
    class FailedRuntime:
        def answer(self, *_args, **_kwargs):
            raise ValueError("secret token from provider")

    engine, db = _db()
    try:
        service = RunService(qa_runtime=FailedRuntime(), registry=Registry())
        service.create_and_execute(db, user_id=1,
            request=RunCreateRequest(question="question"))
        assert "secret token" not in caplog.text
        assert "type=ValueError" in caplog.text
    finally:
        db.close()
        engine.dispose()


def test_started_callback_failure_is_recorded_for_review():
    engine, db = _db()
    try:
        service = RunService(qa_runtime=Runtime(), registry=Registry())

        def fail(_run_id):
            raise RuntimeError("callback failed")

        result = service.create_and_execute(db, user_id=1,
            request=RunCreateRequest(question="question"), on_started=fail)
        assert result.status == RunStatus.FAILED
        assert db.query(ReviewTicketRecord).filter_by(run_id=result.run_id).count() == 1
    finally:
        db.close()
        engine.dispose()


def test_cancel_pending_run_is_terminal_and_does_not_execute():
    engine, db = _db()
    try:
        service = RunService(qa_runtime=Runtime(), registry=Registry())
        record = RunRecord(id="run-cancel", user_id=1, status="pending", question="q",
            model_snapshot_json=Registry().snapshot().model_dump_json())
        db.add(record)
        db.commit()
        result = service.cancel(db, run_id="run-cancel", user_id=1)
        assert result.status == RunStatus.CANCELLED
        assert db.get(RunRecord, "run-cancel").finished_at is not None
        assert service.cancel(db, run_id="run-cancel", user_id=1).status == RunStatus.CANCELLED
        assert [event.event for event in db.query(RunEventRecord).filter_by(run_id="run-cancel")] == ["cancelled"]
    finally:
        db.close()
        engine.dispose()


def test_cancel_during_inference_discards_answer_and_emits_one_terminal_event():
    engine, db = _db()
    try:
        service = RunService(qa_runtime=None, registry=Registry())

        class CancelDuringAnswer:
            def answer(self, _question, run_id, knowledge_snapshot=None):
                assert service.cancel(db, run_id=run_id, user_id=1).status == RunStatus.RUNNING
                return {"answer": "must not persist", "evidence": []}

        service.qa_runtime = CancelDuringAnswer()
        result = service.create_and_execute(db, user_id=1,
            request=RunCreateRequest(question="question"))
        assert result.status == RunStatus.CANCELLED
        assert db.get(RunRecord, result.run_id).answer is None
        assert [event.event for event in db.query(RunEventRecord).filter_by(run_id=result.run_id)] == ["started", "cancelled"]
    finally:
        db.close()
        engine.dispose()


def test_cancel_after_started_event_prevents_inference():
    engine, db = _db()
    try:
        runtime = Runtime()
        service = RunService(qa_runtime=runtime, registry=Registry())
        result = service.create_and_execute(db, user_id=1,
            request=RunCreateRequest(question="question"),
            on_started=lambda run_id: service.cancel(db, run_id=run_id, user_id=1))
        assert result.status == RunStatus.CANCELLED
        assert [event.event for event in db.query(RunEventRecord).filter_by(run_id=result.run_id)] == ["started", "cancelled"]
    finally:
        db.close()
        engine.dispose()


def test_recovery_preserves_history_and_reviews_interrupted_paid_run():
    engine, db = _db()
    try:
        service = RunService(qa_runtime=Runtime(), registry=Registry())
        db.add(RunRecord(id="run-old", user_id=1, status="completed", question="old",
            answer="historical answer", model_snapshot_json=Registry().snapshot().model_dump_json()))
        db.add(RunRecord(id="run-interrupted", user_id=1, status="running", question="q",
            started_at=datetime.now(), model_snapshot_json=ModelSnapshot(
                snapshot_id="paid", profile_name="paid", provider="ark").model_dump_json()))
        db.commit()
        service.recover_interrupted_runs(db)
        assert db.get(RunRecord, "run-old").answer == "historical answer"
        assert db.get(RunRecord, "run-interrupted").status == "failed"
        assert db.query(ReviewTicketRecord).filter_by(run_id="run-interrupted").count() == 1
        assert [event.event for event in db.query(RunEventRecord).filter_by(run_id="run-interrupted")] == ["error"]
    finally:
        db.close()
        engine.dispose()


def test_recovery_completes_local_graph_from_checkpoint(monkeypatch):
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "1")

    class RecoverableRuntime:
        def resume_graph(self, question, run_id, knowledge_snapshot=None):
            assert (question, run_id, knowledge_snapshot) == ("q", "run-local", None)
            return {"answer": "recovered answer", "evidence": [], "verified": False,
                    "agent_mode": "graph", "trace": [{"node": "finalize", "decision": "done"}]}

    engine, db = _db()
    try:
        db.add(RunRecord(id="run-local", user_id=1, status="running", question="q",
            model_snapshot_json=Registry().snapshot().model_dump_json(),
            checkpoint_json='{"node":"retrieve"}'))
        db.commit()
        service = RunService(qa_runtime=RecoverableRuntime(), registry=Registry())
        assert service.recover_interrupted_runs(db) == 1
        record = db.get(RunRecord, "run-local")
        assert record.status == "completed"
        assert record.answer == "recovered answer"
        assert [event.event for event in db.query(RunEventRecord).filter_by(run_id="run-local")] == ["done"]
    finally:
        db.close()
        engine.dispose()


def test_recovery_expired_run_never_invokes_model():
    class NoCallRuntime:
        def resume_graph(self, *_args, **_kwargs):
            raise AssertionError("expired execution must not resume")

    engine, db = _db()
    try:
        db.add(RunRecord(id="expired", user_id=1, status="running", question="q",
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
            model_snapshot_json=Registry().snapshot().model_dump_json(), checkpoint_json='{}'))
        db.commit()
        RunService(qa_runtime=NoCallRuntime(), registry=Registry()).recover_interrupted_runs(db)
        assert db.get(RunRecord, "expired").error_code == "run_timeout"
        assert db.get(RunRecord, "expired").answer is None
    finally:
        db.close()
        engine.dispose()


def test_cancel_during_recovery_blocks_next_node_and_discards_answer():
    from src.system.execution_control import check_execution
    engine, db = _db()
    service = RunService(qa_runtime=None, registry=Registry())

    class CancellingRuntime:
        def resume_graph(self, _question, run_id, **_kwargs):
            check_execution()
            service.cancel(db, run_id=run_id, user_id=1)
            check_execution()
            raise AssertionError("cancelled recovery must not reach the next paid node")

    try:
        service.qa_runtime = CancellingRuntime()
        db.add(RunRecord(id="cancel-recovery", user_id=1, status="running", question="q",
            model_snapshot_json=Registry().snapshot().model_dump_json(), checkpoint_json='{}'))
        db.commit()
        service.recover_interrupted_runs(db)
        assert db.get(RunRecord, "cancel-recovery").status == "cancelled"
        assert db.get(RunRecord, "cancel-recovery").answer is None
        assert db.query(RunEventRecord).filter_by(run_id="cancel-recovery", event="cancelled").count() == 1
    finally:
        db.close()
        engine.dispose()
