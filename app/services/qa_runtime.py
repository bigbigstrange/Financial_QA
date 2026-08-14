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
import sys
import threading
import time
import traceback
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
        self._qa = None
        self._ready = False
        self._loading = False
        self._error: Optional[str] = None
        self._profile_name: Optional[str] = None
        self._last_infer_at: float = 0.0
        self._idle_thread_started = False

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
            if (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("FINANCE_API_KEY") or "").strip():
                os.environ.setdefault("FINANCE_GEN_BACKEND", "api")
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

    def status(self) -> dict[str, Any]:
        s = get_settings()
        mem = memory_guard.snapshot()
        chunks = 0
        if self._qa is not None:
            chunks = len(getattr(self._qa, "knowledge_chunks", []) or [])
        return {
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

    def ensure_loaded(self, profile_name: Optional[str] = None) -> None:
        with self._lock:
            target = profile_name or model_registry.active_profile_name()
            if self._ready and self._qa is not None and self._profile_name == target:
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
                    "mmd_directory": s.kb_path,
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
                self._ready = True
                self._start_idle_watcher()
            except Exception as e:
                self._ready = False
                self._qa = None
                self._error = f"{e}\n{traceback.format_exc()}"
                raise
            finally:
                self._loading = False

    def reload(self, profile_name: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            self._ready = False
            self._drop_engine_unlocked()
        self.ensure_loaded(profile_name)
        return self.status()

    def answer(self, question: str, question_id: str) -> dict[str, Any]:
        self.ensure_loaded()
        ok, snap, msg = memory_guard.allow_infer()
        if not ok:
            raise MemoryError(msg)

        acquired = self._sem().acquire(blocking=False)
        if not acquired:
            raise RuntimeError("推理繁忙：已达并发上限，请稍后重试（防 OOM）")
        try:
            assert self._qa is not None
            # 推理前再检查一次硬限制
            ok2, _, msg2 = memory_guard.allow_infer()
            if not ok2:
                raise MemoryError(msg2)
            result = self._qa.answer_question(question, question_id)
            self._last_infer_at = time.time()
            result["_memory"] = memory_guard.snapshot().to_dict()
            result["_profile"] = self._profile_name
            return result
        finally:
            self._sem().release()

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
        memory_guard.release_python_and_cuda()

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
