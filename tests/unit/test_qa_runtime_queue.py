import threading
import time

import pytest

from app.services.qa_runtime import QARuntime


def test_second_request_waits_for_first_instead_of_busy(monkeypatch):
    runtime = QARuntime()
    runtime._infer_sem = threading.Semaphore(1)
    runtime._ready = True
    entered = threading.Event()
    release = threading.Event()
    completed = []

    class QA:
        def answer_question(self, question, _run_id):
            if question == "first":
                entered.set()
                assert release.wait(3)
            return {"answer": question}

    runtime._qa = QA()
    monkeypatch.setattr(runtime, "ensure_loaded", lambda **_kw: None)
    monkeypatch.setattr(runtime, "agent_mode", lambda: "legacy")
    monkeypatch.setattr("app.services.qa_runtime.memory_guard.allow_infer", lambda: (True, None, ""))
    monkeypatch.setattr("app.services.qa_runtime.memory_guard.snapshot", lambda: type("S", (), {"to_dict": lambda self: {}})())

    first = threading.Thread(target=lambda: completed.append(runtime.answer("first", "1")))
    second = threading.Thread(target=lambda: completed.append(runtime.answer("second", "2")))
    first.start()
    assert entered.wait(2)
    second.start()
    time.sleep(0.1)
    assert second.is_alive()
    release.set()
    first.join(3)
    second.join(3)
    assert sorted(item["answer"] for item in completed) == ["first", "second"]


def test_queue_is_bounded_to_one_waiter(monkeypatch):
    runtime = QARuntime()
    runtime._infer_sem = threading.Semaphore(1)
    runtime._ready = True
    entered = threading.Event()
    release = threading.Event()

    class QA:
        def answer_question(self, question, _run_id):
            if question == "first":
                entered.set()
                assert release.wait(3)
            return {"answer": question}

    runtime._qa = QA()
    monkeypatch.setattr(runtime, "ensure_loaded", lambda **_kw: None)
    monkeypatch.setattr(runtime, "agent_mode", lambda: "legacy")
    monkeypatch.setattr("app.services.qa_runtime.memory_guard.allow_infer", lambda: (True, None, ""))
    monkeypatch.setattr("app.services.qa_runtime.memory_guard.snapshot", lambda: type("S", (), {"to_dict": lambda self: {}})())
    first = threading.Thread(target=lambda: runtime.answer("first", "1"))
    second = threading.Thread(target=lambda: runtime.answer("second", "2"))
    first.start()
    assert entered.wait(2)
    second.start()
    time.sleep(0.1)
    try:
        with pytest.raises(RuntimeError, match="队列已满"):
            runtime.answer("third", "3")
    finally:
        release.set()
        first.join(3)
        second.join(3)
