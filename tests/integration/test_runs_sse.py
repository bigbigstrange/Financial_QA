import json
from contextlib import nullcontext
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import runs
from app.core.deps import get_current_user
from app.db.session import get_db
from src.contracts import RunStatus


def _app(auth=False):
    app = FastAPI()
    app.include_router(runs.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: None
    if auth:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    return app


def _events(text):
    return [(line.split(": ", 1)[1], json.loads(data.split(": ", 1)[1]))
            for line, data in zip(text.splitlines()[::3], text.splitlines()[1::3])]


def test_stream_requires_auth():
    with TestClient(_app()) as client:
        response = client.post("/api/runs/stream", json={"question": "基金认购"})
    assert response.status_code == 401


def test_stream_emits_live_ordered_redacted_nodes_and_one_terminal(monkeypatch):
    class FakeService:
        def create_and_execute(self, db, *, user_id, request, on_started=None, on_node=None):
            assert user_id == 7
            on_started("run-1")
            on_node({"node": "retrieve", "decision": "2 条证据", "elapsed_ms": 3,
                     "detail": {"prompt": "secret"}})
            on_node({"node": "generate", "decision": "完成", "elapsed_ms": 4})
            return SimpleNamespace(status=RunStatus.COMPLETED, run_id="run-1",
                                   verified=False, error_code=None, error_message=None)
    monkeypatch.setattr(runs, "run_service", FakeService())
    monkeypatch.setattr(runs, "SessionLocal", lambda: nullcontext(object()))
    with TestClient(_app(auth=True)) as client:
        response = client.post("/api/runs/stream", json={"question": "基金认购"})
    assert response.status_code == 200
    events = _events(response.text)
    assert [event for event, _ in events] == ["started", "node", "node", "done"]
    assert events[0][1]["run_id"] == "run-1"
    assert events[-1][1]["verified"] is False
    assert "secret" not in response.text
    assert "基金认购" not in response.text


def test_budget_stop_is_one_redacted_terminal_error(monkeypatch):
    class FakeService:
        def create_and_execute(self, db, *, user_id, request, on_started=None, on_node=None):
            on_started("run-2")
            return SimpleNamespace(status=RunStatus.FAILED, run_id="run-2", verified=False,
                                   error_code=SimpleNamespace(value="model_unavailable"),
                                   error_message="internal secret")
    monkeypatch.setattr(runs, "run_service", FakeService())
    monkeypatch.setattr(runs, "SessionLocal", lambda: nullcontext(object()))
    monkeypatch.setattr(runs, "_safe_pilot_stop_reason", lambda: "http_402", raising=False)
    with TestClient(_app(auth=True)) as client:
        response = client.post("/api/runs/stream", json={"question": "基金认购"})
    assert [event for event, _ in _events(response.text)] == ["started", "error"]
    assert _events(response.text)[-1][1]["reason"] == "http_402"
    assert "internal secret" not in response.text


def test_real_graph_stream_nodes_match_final_trace():
    from src.agent.graph import build_graph, run_agent

    class Retriever:
        def search(self, question, top_k=12):
            return [{"chunk_id": "ck_1", "content": "基金认购说明。", "source_doc": "说明.pdf",
                     "page_start": 1, "fusion_score": 0.9}]

    events = []
    result = run_agent(build_graph(Retriever(), llm=None, reranker=None, log=lambda _: None),
                       "基金认购是什么？", on_node=events.append)
    assert [entry["node"] for entry in events] == [entry["node"] for entry in result["trace"]]
    assert "retrieve" in [entry["node"] for entry in events]
    assert all("detail" not in entry for entry in events)
