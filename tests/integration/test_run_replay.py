from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import runs
from app.core.deps import get_current_user
from app.db.base import Base
from app.db.session import get_db
from app.models import RunRecord, ReviewApprovalRecord, ReviewTicketRecord, RunEventRecord
from app.services.run_service import RunService
from src.contracts import ModelSnapshot


class Registry:
    def snapshot(self):
        return ModelSnapshot(snapshot_id="test", profile_name="test", provider="local")


def test_reconnect_replays_only_owned_events_after_last_id(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    service = RunService(qa_runtime=object(), registry=Registry())
    with Session() as db:
        record = RunRecord(id="r1", user_id=7, status="completed", question="secret",
            answer="answer", model_snapshot_json=Registry().snapshot().model_dump_json())
        db.add(record)
        db.commit()
        service._event(db, record, "started", {"run_id": "r1"})
        service._event(db, record, "done", {"run_id": "r1", "verified": True})
        db.commit()
        first_id = service.list_events(db, run_id="r1", user_id=7)[0].id
    monkeypatch.setattr(runs, "run_service", service)
    monkeypatch.setattr(runs, "SessionLocal", Session)
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    user = SimpleNamespace(id=7)
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        response = client.get("/api/runs/r1/events", headers={"Last-Event-ID": str(first_id)})
        assert response.status_code == 200
        assert "event: done" in response.text
        assert "event: started" not in response.text
        assert "id: " in response.text
        user.id = 8
        assert client.get("/api/runs/r1/events").status_code == 404
    engine.dispose()


def test_cancel_endpoint_requires_ownership(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        db.add(RunRecord(id="r1", user_id=7, status="pending", question="q",
            model_snapshot_json=Registry().snapshot().model_dump_json()))
        db.commit()
    monkeypatch.setattr(runs, "run_service", RunService(qa_runtime=object(), registry=Registry()))
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    user = SimpleNamespace(id=8)
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        assert client.post("/api/runs/r1/cancel").status_code == 404
        user.id = 7
        response = client.post("/api/runs/r1/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
    engine.dispose()


def test_post_stream_ids_replay_from_same_durable_log(monkeypatch):
    class Runtime:
        def answer(self, question, run_id, knowledge_snapshot=None, on_node=None):
            on_node({"node": "retrieve", "decision": "found", "elapsed_ms": 1})
            return {"answer": "answer", "evidence": [], "verified": True}

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(runs, "run_service", RunService(qa_runtime=Runtime(), registry=Registry()))
    monkeypatch.setattr(runs, "SessionLocal", Session)
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    with TestClient(app) as client:
        response = client.post("/api/runs/stream", json={"question": "secret"})
        assert response.status_code == 200
        assert [line for line in response.text.splitlines() if line.startswith("event:")] == [
            "event: started", "event: node", "event: done"]
        ids = [int(line[4:].strip()) for line in response.text.splitlines() if line.startswith("id:")]
        assert ids == sorted(set(ids)) and len(ids) == 3
        with Session() as db:
            run_id = db.query(RunRecord.id).one()[0]
        replay = client.get(f"/api/runs/{run_id}/events",
                            headers={"Last-Event-ID": str(ids[0])})
        assert "event: started" not in replay.text
        assert "event: node" in replay.text and "event: done" in replay.text
    engine.dispose()


def test_review_tickets_are_admin_only_and_paginated():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        for index in range(3):
            run_id = f"run-{index}"
            db.add(RunRecord(id=run_id, user_id=7, status="failed", question="secret question",
                error_code="internal_error", answer="secret answer",
                model_snapshot_json=Registry().snapshot().model_dump_json()))
            db.add(ReviewTicketRecord(id=f"ticket-{index}", run_id=run_id,
                user_id=7, reason="failed"))
        db.commit()
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    user = SimpleNamespace(id=7, role="user")
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        assert client.get("/api/runs/review-tickets").status_code == 403
        user.role = "admin"
        first = client.get("/api/runs/review-tickets?limit=2&offset=0")
        second = client.get("/api/runs/review-tickets?limit=2&offset=2")
        assert first.status_code == second.status_code == 200
        assert first.json()["total"] == second.json()["total"] == 3
        assert len(first.json()["items"]) == 2
        assert len(second.json()["items"]) == 1
        assert {item["id"] for item in first.json()["items"]}.isdisjoint(
            {item["id"] for item in second.json()["items"]})
        assert "secret question" not in first.text
        assert "secret answer" not in first.text
    engine.dispose()


def test_review_requires_owner_confirmation_and_is_idempotent(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        db.add(RunRecord(id="review-run", user_id=7, status="completed", question="q",
            answer="a", model_snapshot_json=Registry().snapshot().model_dump_json()))
        db.add(ReviewTicketRecord(id="suggestion", run_id="review-run", user_id=7,
            reason="uncertain", status="suggested"))
        db.commit()
    monkeypatch.setattr(runs, "run_service", RunService(qa_runtime=object(), registry=Registry()))
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    user = SimpleNamespace(id=8)
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        endpoint = "/api/runs/review-run/approvals"
        assert client.post(endpoint, json={"decision": "accept"}).status_code == 404
        user.id = 7
        first = client.post(endpoint, json={"decision": "accept"})
        second = client.post(endpoint, json={"decision": "accept"})
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json() == {"run_id": "review-run",
            "decision": "accept", "ticket_id": "suggestion", "ticket_status": "open"}
        assert client.post(endpoint, json={"decision": "decline"}).status_code == 409
    with Session() as db:
        assert db.query(ReviewApprovalRecord).filter_by(run_id="review-run").count() == 1
        assert db.query(ReviewTicketRecord).filter_by(run_id="review-run").count() == 1
        assert db.query(RunEventRecord).filter_by(run_id="review-run", event="review_decision").count() == 1
    engine.dispose()


def test_review_decline_does_not_open_ticket(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        db.add(RunRecord(id="decline-run", user_id=7, status="completed", question="q",
            model_snapshot_json=Registry().snapshot().model_dump_json()))
        db.commit()
    monkeypatch.setattr(runs, "run_service", RunService(qa_runtime=object(), registry=Registry()))
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    with TestClient(app) as client:
        response = client.post("/api/runs/decline-run/approvals", json={"decision": "decline"})
        assert response.status_code == 200
        assert response.json()["ticket_id"] is None
    with Session() as db:
        assert db.query(ReviewTicketRecord).filter_by(run_id="decline-run").count() == 0
    engine.dispose()


def test_stream_run_can_be_found_by_owner_idempotency_key(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        db.add(RunRecord(id="key-run", user_id=7, status="completed", question="q",
            answer="a", idempotency_key="unique-key-1234",
            model_snapshot_json=Registry().snapshot().model_dump_json()))
        db.commit()
    monkeypatch.setattr(runs, "run_service", RunService(qa_runtime=object(), registry=Registry()))
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    user = SimpleNamespace(id=8)
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        path = "/api/runs/by-key/unique-key-1234"
        assert client.get(path).status_code == 404
        user.id = 7
        result = client.get(path)
        assert result.status_code == 200
        assert result.json()["run_id"] == "key-run"
        assert client.get("/api/runs/by-key/short").status_code == 404
    engine.dispose()
