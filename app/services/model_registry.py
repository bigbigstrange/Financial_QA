# -*- coding: utf-8 -*-
"""
模型配置注册表：支撑「候选 → 评测门禁 → 晋级/回滚」迭代闭环。

落地原则：
- 配置与权重分离：registry 只存 profile/路径/门禁阈值
- 晋级必须有评测记录（eval gate）
- 历史可审计，出问题可一键回滚 active_profile
"""
from __future__ import annotations

import json
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from src.contracts import ModelSnapshot


class ModelRegistry:
    def __init__(self, path: Optional[str] = None) -> None:
        settings = get_settings()
        self.path = Path(path or settings.model_registry_path)
        self._lock = threading.Lock()
        self._ensure()

    def _ensure(self) -> None:
        if self.path.is_file():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "active_profile": "stable-v1",
            "profiles": {
                "stable-v1": {
                    "label": "生产稳定版",
                    "provider": "deepseek",
                    "requested_model": "deepseek-v4-flash",
                    "generation_parameters": {"temperature": 0.2, "max_tokens": 1024},
                    "embedding_model": "BAAI/bge-m3",
                    "prompt_versions": {"answer": "legacy-v1"},
                    "retrieval_config": {"top_k": 12},
                    "parser_config": {"docir": "legacy-v1"},
                    "embed_model": None,
                    "reranker_model": None,
                    "gen_model": None,
                    "image_backend": "visrag",
                    "enable_graph": False,
                    "notes": "default",
                }
            },
            "gates": {
                "min_success_rate": 0.75,
                "max_avg_latency_sec": 45.0,
                "min_avg_confidence": 0.35,
                "min_evidence_coverage": 0.7,
            },
            "history": [],
        }
        self.path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def active_profile_name(self) -> str:
        return str(self.load().get("active_profile") or "stable-v1")

    def get_profile(self, name: Optional[str] = None) -> dict[str, Any]:
        data = self.load()
        key = name or data.get("active_profile")
        profiles = data.get("profiles") or {}
        if key not in profiles:
            raise KeyError(f"未知模型配置 profile: {key}")
        prof = dict(profiles[key])
        prof["name"] = key
        return prof

    def list_profiles(self) -> list[dict[str, Any]]:
        data = self.load()
        active = data.get("active_profile")
        out = []
        for name, prof in (data.get("profiles") or {}).items():
            item = dict(prof)
            item["name"] = name
            item["is_active"] = name == active
            out.append(item)
        return out

    def snapshot(
        self,
        name: Optional[str] = None,
        *,
        evaluation_run_id: Optional[int] = None,
    ) -> ModelSnapshot:
        profile = self.get_profile(name)
        payload = {
            "profile_name": profile["name"],
            "provider": str(profile.get("provider") or "local"),
            "requested_model": profile.get("requested_model") or profile.get("gen_model"),
            "actual_model": profile.get("actual_model") or profile.get("requested_model") or profile.get("gen_model"),
            "generation_parameters": dict(profile.get("generation_parameters") or {}),
            "embedding_model": profile.get("embedding_model") or profile.get("embed_model"),
            "reranker_model": profile.get("reranker_model"),
            "prompt_versions": dict(profile.get("prompt_versions") or {}),
            "retrieval_config": dict(profile.get("retrieval_config") or {}),
            "parser_config": dict(profile.get("parser_config") or {}),
            "evaluation_run_id": evaluation_run_id,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["snapshot_id"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        return ModelSnapshot(**payload)

    def gates(self) -> dict[str, float]:
        return dict(self.load().get("gates") or {})

    def upsert_profile(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        profiles = data.setdefault("profiles", {})
        cur = dict(profiles.get(name) or {})
        for k in (
            "label",
            "provider",
            "requested_model",
            "actual_model",
            "generation_parameters",
            "embedding_model",
            "embed_model",
            "reranker_model",
            "gen_model",
            "prompt_versions",
            "retrieval_config",
            "parser_config",
            "image_backend",
            "enable_graph",
            "notes",
        ):
            if k in payload:
                cur[k] = payload[k]
        profiles[name] = cur
        self._append_history(data, action="upsert_profile", profile=name, detail=payload)
        self.save(data)
        return self.get_profile(name)

    def promote(self, name: str, *, eval_run_id: Optional[int] = None, force: bool = False) -> dict[str, Any]:
        data = self.load()
        if name not in (data.get("profiles") or {}):
            raise KeyError(f"未知 profile: {name}")
        prev = data.get("active_profile")
        if not force and name != prev and eval_run_id is None:
            raise ValueError("晋级生产必须附带通过质量门禁的 eval_run_id")
        if not force and name != prev:
            from app.db.session import SessionLocal
            from app.models import EvalRun

            with SessionLocal() as db:
                reviewed = db.get(EvalRun, eval_run_id)
                if (reviewed is None or reviewed.profile_name != name or
                        reviewed.status != "passed" or reviewed.gate_passed != 1 or
                        not reviewed.report_path):
                    raise ValueError("晋级所用评测不存在、未通过或配置不匹配")
                report = json.loads(Path(reviewed.report_path).read_text(encoding="utf-8"))
                if report.get("quality_gate_passed") is not True:
                    raise ValueError("评测缺少人工标准答案与引用质量门禁，不得晋级")
        data["active_profile"] = name
        self._append_history(
            data,
            action="promote",
            profile=name,
            detail={"from": prev, "eval_run_id": eval_run_id, "force": force},
        )
        self.save(data)
        return self.get_profile(name)

    def rollback(self, to: Optional[str] = None) -> dict[str, Any]:
        data = self.load()
        history = data.get("history") or []
        target = to
        if not target:
            for h in reversed(history):
                if h.get("action") == "promote" and h.get("detail", {}).get("from"):
                    target = h["detail"]["from"]
                    break
        if not target:
            raise ValueError("无可用回滚目标")
        return self.promote(target, force=True)

    def check_gates(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        g = self.gates()
        fails: list[str] = []
        sr = float(metrics.get("success_rate") or 0)
        lat = float(metrics.get("avg_latency_sec") or 9999)
        conf = float(metrics.get("avg_confidence") or 0)
        cov = float(metrics.get("evidence_coverage") or 0)
        if sr < float(g.get("min_success_rate", 0.75)):
            fails.append(f"success_rate {sr} < {g.get('min_success_rate')}")
        if lat > float(g.get("max_avg_latency_sec", 45)):
            fails.append(f"avg_latency_sec {lat} > {g.get('max_avg_latency_sec')}")
        if conf < float(g.get("min_avg_confidence", 0.35)):
            fails.append(f"avg_confidence {conf} < {g.get('min_avg_confidence')}")
        if cov < float(g.get("min_evidence_coverage", 0.7)):
            fails.append(f"evidence_coverage {cov} < {g.get('min_evidence_coverage')}")
        return len(fails) == 0, fails

    @staticmethod
    def _append_history(data: dict[str, Any], *, action: str, profile: str, detail: Any) -> None:
        hist = data.setdefault("history", [])
        hist.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "profile": profile,
                "detail": detail,
            }
        )
        # 保留最近 200 条审计
        data["history"] = hist[-200:]


model_registry = ModelRegistry()
