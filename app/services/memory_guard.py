# -*- coding: utf-8 -*-
"""
进程级内存防护：企业落地中防止「多模型叠载 / 并发推理」导致 OOM。

策略：
1. 按本机物理内存百分比分 warn / soft / hard 三档水位
2. hard：拒绝新推理与模型加载
3. soft：拒绝加载新模型，允许进行中的推理；触发生成模型卸载建议
4. 记录 MemoryEvent，供运维看板与复盘
"""
from __future__ import annotations

import gc
import threading
from dataclasses import asdict, dataclass
from typing import Any, Optional

import psutil

from app.core.config import get_settings


@dataclass
class MemorySnapshot:
    rss_mb: float
    vms_mb: float
    pct: float
    available_mb: float
    total_mb: float
    level: str  # ok | warn | soft | hard
    cuda_allocated_mb: Optional[float] = None
    cuda_reserved_mb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_event: Optional[dict[str, Any]] = None

    def snapshot(self) -> MemorySnapshot:
        settings = get_settings()
        proc = psutil.Process()
        mi = proc.memory_info()
        vm = psutil.virtual_memory()
        rss_mb = mi.rss / (1024 * 1024)
        vms_mb = mi.vms / (1024 * 1024)
        pct = float(vm.percent)
        level = "ok"
        if pct >= settings.mem_hard_pct:
            level = "hard"
        elif pct >= settings.mem_soft_pct:
            level = "soft"
        elif pct >= settings.mem_warn_pct:
            level = "warn"

        cuda_alloc = cuda_res = None
        try:
            import torch

            if torch.cuda.is_available():
                cuda_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_res = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass

        return MemorySnapshot(
            rss_mb=round(rss_mb, 1),
            vms_mb=round(vms_mb, 1),
            pct=round(pct, 1),
            available_mb=round(vm.available / (1024 * 1024), 1),
            total_mb=round(vm.total / (1024 * 1024), 1),
            level=level,
            cuda_allocated_mb=round(cuda_alloc, 1) if cuda_alloc is not None else None,
            cuda_reserved_mb=round(cuda_res, 1) if cuda_res is not None else None,
        )

    def allow_infer(self) -> tuple[bool, MemorySnapshot, str]:
        snap = self.snapshot()
        if snap.level == "hard":
            msg = f"内存硬限制：系统已用 {snap.pct}% ，拒绝新推理以防 OOM"
            self._remember("hard", "reject_infer", snap, msg)
            return False, snap, msg
        if snap.level == "soft":
            self._remember("soft", "allow_infer_caution", snap, "软限制：允许推理但建议卸载空闲模型")
        elif snap.level == "warn":
            self._remember("warn", "memory_warn", snap, "内存告警水位")
        return True, snap, "ok"

    def allow_load_model(self, name: str) -> tuple[bool, MemorySnapshot, str]:
        snap = self.snapshot()
        if snap.level in {"hard", "soft"}:
            msg = f"内存 {snap.level}：拒绝加载模型 {name}（当前 {snap.pct}%）"
            self._remember(snap.level, "reject_load", snap, msg)
            return False, snap, msg
        return True, snap, "ok"

    def release_python_and_cuda(self) -> MemorySnapshot:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        snap = self.snapshot()
        self._remember("info", "release_cache", snap, "gc + cuda empty_cache")
        return snap

    def _remember(self, level: str, action: str, snap: MemorySnapshot, detail: str) -> None:
        self._last_event = {
            "level": level,
            "action": action,
            "rss_mb": snap.rss_mb,
            "pct": snap.pct,
            "detail": detail,
        }

    @property
    def last_event(self) -> Optional[dict[str, Any]]:
        return self._last_event

    def persist_last_event(self, db) -> None:
        """可选：把最近一次事件写入 DB（由调用方注入 Session）。"""
        from app.models import MemoryEvent

        ev = self._last_event
        if not ev:
            return
        with self._lock:
            row = MemoryEvent(
                level=ev["level"],
                action=ev["action"],
                rss_mb=float(ev["rss_mb"]),
                pct=float(ev["pct"]),
                detail=ev.get("detail"),
            )
            db.add(row)
            db.commit()


memory_guard = MemoryGuard()
