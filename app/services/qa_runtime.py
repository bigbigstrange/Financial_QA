# -*- coding: utf-8 -*-
"""
问答运行时：单例引擎 + 并发闸门 + 空闲卸载生成模型 + profile 热切换。

解决企业常见问题：
- 多请求叠载导致内存溢出
- 生成模型常驻占满显存/内存
- 模型迭代后需要安全重建检索栈
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.services.memory_guard import memory_guard
from app.services.model_registry import model_registry

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class QARuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._infer_sem: Optional[threading.Semaphore] = None
        self._queue_slots: Optional[threading.BoundedSemaphore] = None
        self._qa = None
        self._ready = False
        self._loading = False
        self._error: Optional[str] = None
        self._profile_name: Optional[str] = None
        self._knowledge_snapshot_id: Optional[str] = None
        self._manifest_path: Optional[str] = None
        self._last_infer_at: float = 0.0
        self._idle_thread_started = False
        self._graph = None
        self._graph_llm = None
        self._graph_checkpoint_connection: sqlite3.Connection | None = None
        self._graph_checkpoint_context = None
        self._node_journal = None

    # ---- Agent 编排（阶段 C）------------------------------------------------

    @staticmethod
    def agent_mode() -> str:
        """`FINANCE_AGENT_MODE=graph|legacy`（默认 legacy，保证可回退）。

        优先读 `os.environ`（API 启动时由 `app/main.py` 从 .env 载入），
        读不到再回退到 Settings —— 这样脚本与测试环境也能正确取值。
        """
        raw = (os.environ.get("FINANCE_AGENT_MODE") or "").strip().lower()
        if not raw:
            try:
                from app.core.config import get_settings

                raw = str(get_settings().finance_agent_mode or "").strip().lower()
            except Exception:  # noqa: BLE001 — 配置不可用时按 legacy 处理
                raw = ""
        return "graph" if raw == "graph" else "legacy"

    def _build_agent_graph(self):
        """懒装配 LangGraph Agent，复用已加载的检索栈。

        复用 `self._qa.hybrid_retriever` 而非另建索引，避免同一份知识库被加载两遍。
        LLM 不可用时传 None —— 图会退化为「总是检索 + 抽取式作答」，不会失败。
        """
        if self._graph is not None:
            return self._graph
        if self._qa is None:
            raise RuntimeError("引擎未就绪，无法装配 Agent")
        retriever = getattr(self._qa, "hybrid_retriever", None)
        if retriever is None:
            raise RuntimeError("检索栈未就绪，无法装配 Agent")

        from src.agent import build_graph
        from src.agent.nodes import AgentConfig

        llm = None
        try:
            from src.system.llm_api import DeepSeekAPIClient

            client = DeepSeekAPIClient()
            llm = client if client.enabled else None
        except Exception:  # noqa: BLE001 — LLM 不可用不应阻断装配
            llm = None
        self._graph_llm = llm
        from src.agent.node_journal import NodeJournal
        database_url = get_settings().database_url
        if database_url.startswith("postgresql"):
            from langgraph.checkpoint.postgres import PostgresSaver
            self._graph_checkpoint_context = PostgresSaver.from_conn_string(
                database_url.replace("postgresql+psycopg://", "postgresql://", 1))
            checkpointer = self._graph_checkpoint_context.__enter__()
            checkpointer.setup()
            self._node_journal = NodeJournal(database_url)
        else:
            from langgraph.checkpoint.sqlite import SqliteSaver
            path = Path(os.environ.get("FINANCE_GRAPH_CHECKPOINT_PATH") or
                        ROOT / "output" / "run_checkpoints.sqlite3")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._graph_checkpoint_connection = sqlite3.connect(path, check_same_thread=False)
            checkpointer = SqliteSaver(self._graph_checkpoint_connection)
            self._node_journal = NodeJournal(f"sqlite:///{path.as_posix()}")
        self._graph = build_graph(retriever, llm, config=AgentConfig.from_env(),
                                  checkpointer=checkpointer,
                                  node_journal=self._node_journal)
        return self._graph

    def _answer_with_graph(self, question: str, question_id: str,
                           on_node=None, *, use_paid_model: bool = True,
                           use_web_search: bool = True) -> dict[str, Any]:
        """跑一次 LangGraph Agent，并把结果整理成与 legacy 一致的结构。"""
        from src.agent import run_agent
        from src.system.cost_policy import paid_api_disabled

        graph = self._build_agent_graph()
        if self._graph_llm is not None:
            self._graph_llm.last_response = None
        state = run_agent(
            graph,
            question,
            run_id=question_id,
            knowledge_snapshot_id=self._knowledge_snapshot_id or "",
            use_paid_model=use_paid_model and not paid_api_disabled(),
            use_web_search=use_web_search,
            on_node=on_node,
        )
        return self._graph_result_from_state(state, question_id)

    def resume_graph(self, question: str, run_id: str, knowledge_snapshot=None):
        """Resume a matching checkpoint without replaying an uncertain paid call.

        Completed node results are durable. Unknown node outcomes remain fenced;
        code/model changes cannot silently resume an old execution.
        """
        from src.system.cost_policy import paid_api_disabled

        if self.agent_mode() != "graph":
            return None
        manifest_path = knowledge_snapshot.manifest_path if knowledge_snapshot else None
        self.ensure_loaded(manifest_path=manifest_path)
        graph = self._build_agent_graph()
        from src.agent.graph import DEFAULT_RECURSION_LIMIT

        config = {"configurable": {"thread_id": run_id},
                  "recursion_limit": DEFAULT_RECURSION_LIMIT}
        checkpoint = graph.get_state(config)
        state = checkpoint.values or {}
        if (state.get("question") != question or
                state.get("knowledge_snapshot_id", "") !=
                (knowledge_snapshot.snapshot_id if knowledge_snapshot else "")):
            return None
        paid_mode = state.get("use_paid_model", True) and not paid_api_disabled()
        if checkpoint.next:
            if (paid_mode and state.get("execution_signature") !=
                    getattr(graph, "execution_signature", None)):
                return None
            state = graph.invoke(None, config)
        result = self._graph_result_from_state(state, run_id)
        if paid_mode:
            # A restored terminal checkpoint may have no in-process gateway
            # response. Keep the paid model snapshot already bound to the run.
            result.pop("_local_only", None)
        return result

    def _graph_result_from_state(self, state: dict[str, Any], question_id: str) -> dict[str, Any]:
        evidences = (state["relevant_documents"] if "relevant_documents" in state
                     else state.get("documents") or [])
        from src.system.cost_policy import paid_api_disabled
        paused = paid_api_disabled() or not state.get("use_paid_model", True)
        answer = state.get("generation") or ""
        trace = list(state.get("trace") or [])
        confidence = self._derive_confidence(state)
        if paused:
            reason = ("本次已关闭豆包模型" if not state.get("use_paid_model", True)
                      else "服务端付费模型已暂停")
            answer = (f"【本地检索模式】{reason}。以下仅为候选原文摘录，"
                      "未经模型相关性判断或答案校验，可能与问题无关。\n\n") + answer
            confidence = min(confidence, 0.3)
            trace.insert(0, {"node": "runtime_policy", "elapsed_ms": 0,
                "decision": "local_retrieval_only", "detail": {
                    "paid_api_disabled": paid_api_disabled(),
                    "use_paid_model": state.get("use_paid_model", True),
                    "answer_verified": False}})
        result = {
            "question_id": question_id,
            "question": state.get("question") or "",
            "answer": answer,
            "thinking": "",
            "sub_questions": state.get("sub_questions") or [],
            "knowledge_points": list(evidences),
            "evidence": list(evidences),
            "confidence": confidence,
            "agent_mode": "graph",
            "route": state.get("route"),
            "retry_retrieve": int(state.get("retry_retrieve") or 0),
            "retry_generate": int(state.get("retry_generate") or 0),
            "trace": trace,
            "verified": bool(state.get("verified")) and not paused,
        }
        if self._graph_llm is not None and self._graph_llm.last_response is not None:
            result["_model_snapshot"] = self._graph_llm.gateway.snapshot
        else:
            result["_local_only"] = True
        return result

    @staticmethod
    def _derive_confidence(state: dict[str, Any]) -> float:
        """优先用图内收尾节点写入的置信度；缺失时按同一规则重算。"""
        if state.get("confidence") is not None:
            return float(state["confidence"])
        from src.agent.nodes import derive_confidence

        return float(derive_confidence(state))

    def _settings_demo_env(self) -> None:
        s = get_settings()
        if s.finance_web_demo:
            # Windows 演示默认 CPU，降低 CUDA 原生崩溃（无 Python 堆栈直接退出）概率
            os.environ.setdefault("FINANCE_EMBED_DEVICE", "cpu")
            os.environ.setdefault("FINANCE_GEN_DEVICE", "cpu")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault("FINANCE_MAX_BLOCKS", str(min(s.max_blocks, 2000)))
            os.environ.setdefault("FINANCE_SKIP_IMAGE", "1" if s.skip_image else "0")
            os.environ.setdefault("ENABLE_GRAPH", "true" if s.enable_graph else "false")
            os.environ.setdefault("FINANCE_EMBED_BATCH_SIZE", str(min(s.embed_batch_size, 2)))
            os.environ.setdefault("FINANCE_RERANK_BACKEND", "hybrid")
            from src.system.llm_api import configured_api_key
            if configured_api_key():
                os.environ.setdefault("FINANCE_GEN_BACKEND", "api")
                if (os.environ.get("FINANCE_LLM_PROVIDER") or "").strip().lower() != "ark":
                    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")
                os.environ["FINANCE_SKIP_GEN"] = "0"
            else:
                os.environ.setdefault("FINANCE_SKIP_GEN", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            os.environ.setdefault(
                "FINANCE_DEMO_KEYWORDS",
                "掌上银行|网银盾|数字证书|证书在线|客服|浦东发展|浦发|建行|基金|账户|网银|"
                "手机银行|邮政储蓄|邮储|系统升级|95580|40088|交通银行|交行|USBkey|USBKey|"
                "总资产|我的账户|明细查询|激活",
            )

    def _sem(self) -> threading.Semaphore:
        if self._infer_sem is None:
            self._infer_sem = threading.Semaphore(get_settings().max_concurrent_infer)
        return self._infer_sem

    def _slots(self) -> threading.BoundedSemaphore:
        with self._lock:
            if self._queue_slots is None:
                self._queue_slots = threading.BoundedSemaphore(
                    get_settings().max_concurrent_infer + 1
                )
            return self._queue_slots

    def status(self) -> dict[str, Any]:
        from src.system.cost_policy import paid_api_disabled
        s = get_settings()
        mem = memory_guard.snapshot()
        budget_stop_reason = None
        if (os.environ.get("FINANCE_LLM_PROVIDER") or "").lower() == "ark":
            from src.system.pilot_budget import PilotBudget
            budget_path = Path(os.environ.get("FINANCE_PILOT_BUDGET_PATH") or
                               ROOT / "output" / "pilot_budget.sqlite3")
            if budget_path.exists():
                try:
                    budget_stop_reason = PilotBudget(budget_path).summary()["stop_reason"]
                except (OSError, sqlite3.Error, ValueError):
                    budget_stop_reason = "budget_status_unavailable"
        chunks = 0
        text_documents: set[str] = set()
        image_documents: set[str] = set()
        if self._qa is not None:
            blocks = getattr(self._qa, "knowledge_chunks", []) or []
            chunks = len(blocks)
            text_documents = {b.get("document_id") or b.get("source_doc") for b in blocks}
            image_documents = {p.get("document_id") for p in
                (getattr(getattr(self._qa, "image_retriever", None), "pages", []) or [])
                if p.get("document_id")}
        documents = len(text_documents | image_documents)
        if self._qa is not None and self._manifest_path:
            try:
                from src.pipeline.snapshots import read_manifest
                documents = len(read_manifest(Path(self._manifest_path)).versions)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return {
            "paid_api_disabled": paid_api_disabled(),
            "pilot_budget_stop_reason": budget_stop_reason,
            "web_search_configured": bool((os.environ.get("TAVILY_API_KEY") or "").strip()),
            "vision": dict(getattr(getattr(self._qa, "image_retriever", None), "stats", {})),
            "documents": documents,
            "text_documents": len(text_documents),
            "image_only_documents": len(image_documents - text_documents),
            "knowledge_snapshot_id": self._knowledge_snapshot_id,
            "ready": self._ready,
            "loading": self._loading,
            "error": self._error,
            "profile": self._profile_name or model_registry.active_profile_name(),
            "chunks": chunks,
            "demo": s.finance_web_demo,
            "kb_path": s.kb_path,
            "cache_dir": s.cache_dir,
            "memory": mem.to_dict(),
            "max_concurrent_infer": s.max_concurrent_infer,
            "gen_idle_unload_sec": s.gen_idle_unload_sec,
        }

    def ensure_loaded(
        self,
        profile_name: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            target = profile_name or model_registry.active_profile_name()
            if manifest_path is None:
                snapshot_match = self._knowledge_snapshot_id is None
            else:
                snapshot_match = self._manifest_path == manifest_path
            if (
                self._ready
                and self._qa is not None
                and self._profile_name == target
                and snapshot_match
            ):
                return
            if self._loading:
                raise RuntimeError("引擎正在加载，请稍候")
            self._loading = True
            self._error = None
            try:
                self._settings_demo_env()
                ok, snap, msg = memory_guard.allow_load_model("qa_stack")
                if not ok:
                    raise MemoryError(msg)

                from cache_paths import prune_legacy_and_active, resolve_cache_dir
                from src.system.qa_engine import FinancialQASystem

                s = get_settings()
                prof = model_registry.get_profile(target)
                cache = resolve_cache_dir(s.cache_dir)
                prune_legacy_and_active(ROOT / "vector_cache", cache)

                # 切换 profile 时先释放旧引擎
                self._drop_engine_unlocked()

                kwargs: dict[str, Any] = {
                    "mmd_directory": manifest_path or s.kb_path,
                    "cache_dir": str(cache),
                    "kg_dir": s.kg_dir,
                    "enable_graph": bool(prof.get("enable_graph", s.enable_graph)),
                    "image_backend": str(prof.get("image_backend") or s.image_backend),
                }
                if prof.get("embed_model"):
                    kwargs["embed_model"] = prof["embed_model"]
                if prof.get("reranker_model"):
                    kwargs["reranker_model"] = prof["reranker_model"]
                if prof.get("gen_model"):
                    kwargs["gen_model"] = prof["gen_model"]

                qa = FinancialQASystem(**kwargs)
                qa.build_vector_index(force_rebuild=False)
                self._qa = qa
                self._profile_name = target
                self._knowledge_snapshot_id = (
                    None
                    if manifest_path is None
                    else self._snapshot_id_from_path(manifest_path)
                )
                self._manifest_path = manifest_path
                self._ready = True
                self._start_idle_watcher()
            except Exception as e:
                self._ready = False
                self._qa = None
                # Health is public; do not expose local paths, provider responses,
                # document content, or a traceback through /api/health.
                self._error = (str(e) if isinstance(e, MemoryError)
                               else f"引擎加载失败（{type(e).__name__}）")
                raise
            finally:
                self._loading = False

    def reload(self, profile_name: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            self._ready = False
            self._knowledge_snapshot_id = None
            self._manifest_path = None
            self._drop_engine_unlocked()
        self.ensure_loaded(profile_name)
        return self.status()

    def answer(
        self,
        question: str,
        question_id: str,
        knowledge_snapshot=None,
        on_node=None,
        cancel_check=None,
        use_paid_model: bool = True,
        use_web_search: bool = True,
    ) -> dict[str, Any]:
        manifest_path = (
            knowledge_snapshot.manifest_path
            if knowledge_snapshot is not None
            else None
        )
        target_snapshot = (
            knowledge_snapshot.snapshot_id
            if knowledge_snapshot is not None
            else None
        )
        slots = self._slots()
        if not slots.acquire(blocking=False):
            raise RuntimeError("推理队列已满，请稍后重试")
        from src.system.execution_control import execution_guard
        guard_token = execution_guard.set(cancel_check)
        acquired = False
        try:
            while not acquired:
                if cancel_check is not None:
                    cancel_check()
                acquired = self._sem().acquire(timeout=0.1)
            if cancel_check is not None:
                cancel_check()
            if manifest_path is None and self._manifest_path is not None:
                with self._lock:
                    self._drop_engine_unlocked()
                    self._knowledge_snapshot_id = None
                    self._manifest_path = None
            elif manifest_path is not None and self._manifest_path != manifest_path:
                with self._lock:
                    self._drop_engine_unlocked()
                    self._knowledge_snapshot_id = None
                    self._manifest_path = None
            self.ensure_loaded(manifest_path=manifest_path)
            ok, _, msg = memory_guard.allow_infer()
            if not ok:
                raise MemoryError(msg)
            assert self._qa is not None
            # 推理前再检查一次硬限制
            ok2, _, msg2 = memory_guard.allow_infer()
            if not ok2:
                raise MemoryError(msg2)
            if self.agent_mode() == "graph":
                result = self._answer_with_graph(
                    question, question_id, on_node=on_node,
                    use_paid_model=use_paid_model, use_web_search=use_web_search,
                )
            else:
                if not use_paid_model or not use_web_search:
                    raise ValueError("本次问答开关需要 FINANCE_AGENT_MODE=graph")
                result = self._qa.answer_question(question, question_id)
            self._last_infer_at = time.time()
            result["_memory"] = memory_guard.snapshot().to_dict()
            result["_profile"] = self._profile_name
            result["_knowledge_snapshot_id"] = target_snapshot
            return result
        finally:
            execution_guard.reset(guard_token)
            if acquired:
                self._sem().release()
            slots.release()

    def unload_generator(self) -> dict[str, Any]:
        """卸载生成模型，保留检索索引，降低常驻内存。"""
        with self._lock:
            if self._qa is None:
                return {"unloaded": False, "reason": "engine_not_loaded", "memory": memory_guard.snapshot().to_dict()}
            agent = getattr(self._qa, "_agent", None)
            if agent is not None:
                answerer = getattr(agent, "answerer", None)
                if answerer is not None:
                    for attr in ("model", "tokenizer", "_model", "_tokenizer", "pipe"):
                        if hasattr(answerer, attr):
                            try:
                                setattr(answerer, attr, None)
                            except Exception:
                                pass
                # 丢弃整个 agent，下次提问再懒加载
                self._qa._agent = None
            snap = memory_guard.release_python_and_cuda()
            memory_guard._remember("info", "unload_generator", snap, "idle unload generative model")
            return {"unloaded": True, "memory": snap.to_dict()}

    def _drop_engine_unlocked(self) -> None:
        self._qa = None
        self._ready = False
        self._graph = None          # Agent 图持有检索栈引用，必须一并丢弃
        self._graph_llm = None
        if self._graph_checkpoint_connection is not None:
            self._graph_checkpoint_connection.close()
            self._graph_checkpoint_connection = None
        if self._graph_checkpoint_context is not None:
            self._graph_checkpoint_context.__exit__(None, None, None)
            self._graph_checkpoint_context = None
        if self._node_journal is not None:
            self._node_journal.close()
            self._node_journal = None
        memory_guard.release_python_and_cuda()

    @staticmethod
    def _snapshot_id_from_path(manifest_path: str) -> str:
        import json
        from pathlib import Path

        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return str(data.get("snapshot_id") or "")

    def _start_idle_watcher(self) -> None:
        if self._idle_thread_started:
            return
        ttl = get_settings().gen_idle_unload_sec
        if ttl <= 0:
            return
        self._idle_thread_started = True

        def _loop() -> None:
            while True:
                time.sleep(30)
                try:
                    if self._last_infer_at <= 0:
                        continue
                    if time.time() - self._last_infer_at < ttl:
                        continue
                    if self._qa is not None and getattr(self._qa, "_agent", None) is not None:
                        self.unload_generator()
                        self._last_infer_at = 0.0
                except Exception:
                    pass

        threading.Thread(target=_loop, daemon=True, name="gen-idle-unload").start()


qa_runtime = QARuntime()
