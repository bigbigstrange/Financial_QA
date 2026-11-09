from app.services.qa_runtime import QARuntime
from src.agent import run_agent
from src.agent.state import initial_state


def test_graph_checkpoint_survives_runtime_recreation(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_GRAPH_CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite3"))

    class Retriever:
        def search(self, question, top_k=12):
            return [{"chunk_id": "c1", "content": "基金说明", "source_doc": "rules.pdf",
                     "page_start": 1, "fusion_score": 0.9}]

    first = QARuntime()
    first._qa = type("QA", (), {"hybrid_retriever": Retriever()})()
    monkeypatch.setattr("src.system.llm_api.DeepSeekAPIClient", lambda: type("LLM", (), {"enabled": False})())
    graph = first._build_agent_graph()
    result = run_agent(graph, "基金是什么？", run_id="checkpoint-test")
    assert result["trace"]
    first._drop_engine_unlocked()

    second = QARuntime()
    second._qa = type("QA", (), {"hybrid_retriever": Retriever()})()
    restored = second._build_agent_graph().get_state(
        {"configurable": {"thread_id": "checkpoint-test"}})
    assert restored.values["question"] == "基金是什么？"
    assert restored.next == ()
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "1")
    monkeypatch.setenv("FINANCE_AGENT_MODE", "graph")
    monkeypatch.setattr(second, "ensure_loaded", lambda **_kw: None)
    resumed = second.resume_graph("基金是什么？", "checkpoint-test")
    assert resumed["answer"]
    assert resumed["verified"] is False
    second._drop_engine_unlocked()


def test_local_graph_resumes_an_incomplete_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_GRAPH_CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "1")
    monkeypatch.setenv("FINANCE_AGENT_MODE", "graph")

    class Retriever:
        def search(self, question, top_k=12):
            return [{"chunk_id": "c1", "content": "基金说明", "source_doc": "rules.pdf",
                     "page_start": 1, "fusion_score": 0.9}]

    monkeypatch.setattr("src.system.llm_api.DeepSeekAPIClient", lambda: type("LLM", (), {"enabled": False})())
    first = QARuntime()
    first._qa = type("QA", (), {"hybrid_retriever": Retriever()})()
    graph = first._build_agent_graph()
    config = {"configurable": {"thread_id": "incomplete-run"}}
    stream = graph.stream(initial_state("基金是什么？", run_id="incomplete-run"),
                          config, stream_mode="values")
    next(stream)
    next(stream)
    stream.close()
    assert graph.get_state(config).next
    first._drop_engine_unlocked()

    second = QARuntime()
    second._qa = type("QA", (), {"hybrid_retriever": Retriever()})()
    monkeypatch.setattr(second, "ensure_loaded", lambda **_kw: None)
    recovered = second.resume_graph("基金是什么？", "incomplete-run")
    assert recovered["answer"]
    assert second._build_agent_graph().get_state(config).next == ()
    second._drop_engine_unlocked()


def test_paid_mode_recovers_terminal_but_rejects_unversioned_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_GRAPH_CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setenv("FINANCE_AGENT_MODE", "graph")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")

    class Retriever:
        def search(self, question, top_k=12):
            return [{"chunk_id": "c1", "content": "基金说明", "source_doc": "rules.pdf",
                     "page_start": 1, "fusion_score": 0.9}]

    monkeypatch.setattr("src.system.llm_api.DeepSeekAPIClient",
                        lambda: type("LLM", (), {"enabled": False})())
    first = QARuntime()
    first._qa = type("QA", (), {"hybrid_retriever": Retriever()})()
    graph = first._build_agent_graph()
    run_agent(graph, "基金是什么？", run_id="terminal-paid")
    config = {"configurable": {"thread_id": "incomplete-paid"}}
    stream = graph.stream(initial_state("基金是什么？", run_id="incomplete-paid"),
                          config, stream_mode="values")
    next(stream)
    next(stream)
    stream.close()
    assert graph.get_state(config).next
    first._drop_engine_unlocked()

    second = QARuntime()
    second._qa = type("QA", (), {"hybrid_retriever": Retriever()})()
    monkeypatch.setattr(second, "ensure_loaded", lambda **_kw: None)
    restored = second.resume_graph("基金是什么？", "terminal-paid")
    assert restored["answer"]
    assert "_local_only" not in restored
    assert second.resume_graph("基金是什么？", "incomplete-paid") is None
    second._drop_engine_unlocked()


def test_paid_resume_reuses_result_saved_before_graph_checkpoint(tmp_path, monkeypatch):
    import json
    import pytest

    monkeypatch.setenv("FINANCE_GRAPH_CHECKPOINT_PATH", str(tmp_path / "nodes.sqlite3"))
    monkeypatch.setenv("FINANCE_AGENT_MODE", "graph")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    calls = []

    class LLM:
        enabled, provider, model, last_response = True, "ark", "fixed-model", None
        def chat(self, messages, **kwargs):
            calls.append(kwargs["node"])
            return (json.dumps({"route": "direct"}) if kwargs["node"] == "route_query"
                    else "你好")

    monkeypatch.setattr("src.system.llm_api.DeepSeekAPIClient", LLM)
    first = QARuntime()
    first._qa = type("QA", (), {"hybrid_retriever": object()})()
    graph = first._build_agent_graph()
    original = first._node_journal.execute
    def crash_after_save(*args):
        result = original(*args)
        if args[1] == "route_query":
            raise SystemExit("simulated process exit after saved result")
        return result
    monkeypatch.setattr(first._node_journal, "execute", crash_after_save)
    with pytest.raises(SystemExit):
        run_agent(graph, "你好", run_id="paid-gap")
    first._drop_engine_unlocked()
    second = QARuntime()
    second._qa = type("QA", (), {"hybrid_retriever": object()})()
    monkeypatch.setattr(second, "ensure_loaded", lambda **_kw: None)
    result = second.resume_graph("你好", "paid-gap")
    assert result["answer"] == "你好"
    assert calls == ["route_query", "direct_answer"]
    second._drop_engine_unlocked()


def test_paid_resume_fences_unknown_node_outcome(tmp_path, monkeypatch):
    import pytest
    from src.contracts import ContractError

    monkeypatch.setenv("FINANCE_GRAPH_CHECKPOINT_PATH", str(tmp_path / "unknown.sqlite3"))
    monkeypatch.setenv("FINANCE_AGENT_MODE", "graph")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    calls = []
    class LLM:
        enabled, provider, model, last_response = True, "ark", "fixed-model", None
        def chat(self, *_args, **_kwargs):
            calls.append(1)
            raise SystemExit("simulated exit during model request")
    monkeypatch.setattr("src.system.llm_api.DeepSeekAPIClient", LLM)
    first = QARuntime()
    first._qa = type("QA", (), {"hybrid_retriever": object()})()
    with pytest.raises(SystemExit):
        run_agent(first._build_agent_graph(), "问题", run_id="unknown-paid")
    first._drop_engine_unlocked()
    second = QARuntime()
    second._qa = type("QA", (), {"hybrid_retriever": object()})()
    monkeypatch.setattr(second, "ensure_loaded", lambda **_kw: None)
    with pytest.raises(ContractError, match="结果不确定"):
        second.resume_graph("问题", "unknown-paid")
    assert calls == [1]
    second._drop_engine_unlocked()


def test_real_process_exit_recovers_saved_node_without_replaying(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time

    monkeypatch.setenv("FINANCE_GRAPH_CHECKPOINT_PATH", str(tmp_path / "process.sqlite3"))
    monkeypatch.setenv("FINANCE_AGENT_MODE", "graph")
    monkeypatch.setenv("FINANCE_DISABLE_PAID_API", "0")
    code = '''
import os,json
from unittest.mock import patch
from app.services.qa_runtime import QARuntime
from src.agent import run_agent
class LLM:
 enabled,provider,model,last_response=True,"ark","fixed-process-model",None
 def chat(self,messages,**kwargs): return json.dumps({"route":"direct"})
with patch("src.system.llm_api.DeepSeekAPIClient",LLM):
 runtime=QARuntime()
 runtime._qa=type("QA",(),{"hybrid_retriever":object()})()
 graph=runtime._build_agent_graph()
 original=runtime._node_journal.execute
 def crash(*args):
  original(*args)
  os._exit(73)
 runtime._node_journal.execute=crash
 run_agent(graph,"你好",run_id="__RUN_ID__")
'''

    def crashed_child(run_id: str) -> None:
        child = subprocess.run(
            [sys.executable, "-c", code.replace("__RUN_ID__", run_id)],
            timeout=30, capture_output=True,
        )
        assert child.returncode == 73, child.stderr.decode(errors="replace")

    def resume_after_crash(run_id: str):
        calls = []

        class LLM:
            enabled, provider, model, last_response = True, "ark", "fixed-process-model", None

            def chat(self, messages, **kwargs):
                calls.append(kwargs["node"])
                return "你好"

        with monkeypatch.context() as mp:
            mp.setattr("src.system.llm_api.DeepSeekAPIClient", LLM)
            runtime = QARuntime()
            runtime._qa = type("QA", (), {"hybrid_retriever": object()})()
            mp.setattr(runtime, "ensure_loaded", lambda **_kw: None)
            try:
                result = None
                # 子进程用 os._exit 模拟崩溃，checkpoint 对新连接存在毫秒级可见性延迟，
                # 这里做有界轮询；不影响"已完成节点不重放"的断言。
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    result = runtime.resume_graph("你好", run_id)
                    if result is not None:
                        break
                    time.sleep(0.05)
                return result, calls
            finally:
                runtime._drop_engine_unlocked()

    # 极端情况下（CI 磁盘慢）第一次可能读不到 checkpoint，用全新 run_id 重试整个场景。
    result, calls = None, []
    for attempt in range(3):
        run_id = f"real-process-gap-{attempt}"
        crashed_child(run_id)
        result, calls = resume_after_crash(run_id)
        # 只有真正拿到答案才算恢复成功；否则换新 run_id 重跑整个场景
        if result is not None and result.get("answer"):
            break

    assert result is not None, "checkpoint 未被恢复：子进程写入与父进程读取之间存在可见性竞态"
    assert result["answer"] == "你好"
    assert calls == ["direct_answer"]



