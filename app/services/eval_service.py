# -*- coding: utf-8 -*-
"""
持续评测服务：模型迭代的「可量化门禁」。

流程：候选 profile → 跑 golden set → 写 EvalRun + JSON 报告 → 门禁判定 → 可选 promote
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import EvalRun
from app.services.model_registry import model_registry
from app.services.qa_runtime import qa_runtime
from scripts.quality_baseline import (DEFAULT_CHUNKS, DEFAULT_MANIFEST,
                                      DEFAULT_REVIEW, DEFAULT_WORKBOOK, validate)


class EvalService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def run_sync(
        self,
        db: Session,
        *,
        profile_name: Optional[str] = None,
        max_samples: int = 20,
        promote_if_pass: bool = False,
    ) -> EvalRun:
        if self._running:
            raise RuntimeError("已有评测任务在执行")
        settings = get_settings()
        if settings.app_env == "production":
            raise ValueError("生产评测需独立 Worker；禁止占用在线问答实例")
        active_profile = model_registry.active_profile_name()
        if profile_name is not None and profile_name != active_profile:
            raise ValueError("候选模型须在隔离评测进程运行，禁止重载在线问答实例")
        # The supplied workbook has questions but no answers. Never spend on a
        # batch or reload the online runtime before the human gold is ready.
        if not DEFAULT_REVIEW.is_file():
            raise ValueError("缺少人工审核基线；先运行 scripts.quality_baseline prepare")
        quality = validate(DEFAULT_WORKBOOK, DEFAULT_MANIFEST,
                           DEFAULT_CHUNKS, DEFAULT_REVIEW)
        if not quality["quality_gate_ready"]:
            raise ValueError(
                f"人工金标准未就绪：待审核 {quality['pending_cases']} 题，"
                f"结构错误 {len(quality['errors'])} 项；禁止付费批量评测或模型晋级"
            )
        self._running = True
        profile = profile_name or model_registry.active_profile_name()
        run = EvalRun(profile_name=profile, status="running")
        db.add(run)
        db.commit()
        db.refresh(run)

        out_dir = Path(settings.eval_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"eval_{run.id}_{uuid.uuid4().hex[:8]}.json"

        try:
            # Local development only: use the active instance without changing
            # its profile. Candidate A/B belongs in a separate worker process.
            qa_runtime.ensure_loaded(profile)

            golden = Path(settings.eval_golden_path)
            if not golden.is_file():
                raise FileNotFoundError(f"找不到评测集: {golden}")

            df = pd.read_excel(golden)
            if "question" not in df.columns:
                raise ValueError("评测集需含 question 列")
            if max_samples > 0:
                df = df.head(max_samples)

            details = []
            total_lat = 0.0
            success = 0
            conf_sum = 0.0
            evidence_ok = 0

            for idx, row in df.iterrows():
                qid = str(row["question_id"]) if "question_id" in df.columns else f"eval-{idx}"
                q = str(row["question"])
                t0 = time.time()
                try:
                    res = qa_runtime.answer(q, qid)
                    err = None
                except Exception as e:
                    res = {"answer": "", "confidence": 0.0, "evidence": []}
                    err = str(e)
                dt = time.time() - t0
                total_lat += dt
                ev = res.get("evidence") or []
                conf = float(res.get("confidence") or 0.0)
                conf_sum += conf
                ok = bool(ev) and bool((res.get("answer") or "").strip()) and not err
                success += 1 if ok else 0
                evidence_ok += 1 if ev else 0
                details.append(
                    {
                        "question_id": qid,
                        "latency_sec": round(dt, 4),
                        "confidence": conf,
                        "evidence_count": len(ev),
                        "ok": ok,
                        "error": err,
                    }
                )

            n = max(len(details), 1)
            metrics = {
                "samples": len(details),
                "success_rate": round(success / n, 4),
                "avg_latency_sec": round(total_lat / n, 4),
                "avg_confidence": round(conf_sum / n, 4),
                "evidence_coverage": round(evidence_ok / n, 4),
            }
            _, fails = model_registry.check_gates(metrics)
            # Operational completion is not factual accuracy. The reviewed
            # answer-scoring stage must be implemented and independently
            # accepted before this legacy endpoint may promote a profile.
            fails.append("尚无逐题事实正确与引用支持的独立评分，禁止晋级")
            passed = False
            report = {
                "eval_run_id": run.id,
                "profile": profile,
                "metrics": metrics,
                "gate_passed": passed,
                "gate_failures": fails,
                "details": details,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            run.status = "passed" if passed else "failed"
            run.samples = metrics["samples"]
            run.success_rate = metrics["success_rate"]
            run.avg_latency_sec = metrics["avg_latency_sec"]
            run.avg_confidence = metrics["avg_confidence"]
            run.evidence_coverage = metrics["evidence_coverage"]
            run.gate_passed = 1 if passed else 0
            run.report_path = str(report_path)
            run.summary_json = json.dumps({"gate_failures": fails, "metrics": metrics}, ensure_ascii=False)
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(run)

            return run
        except Exception as e:
            run.status = "error"
            run.summary_json = json.dumps({"error": str(e)}, ensure_ascii=False)
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(run)
            raise
        finally:
            self._running = False

    def start_async(self, SessionLocal, **kwargs: Any) -> int:
        """后台线程跑评测；先落 queued 占位，再异步写出正式结果。"""
        if self._running:
            raise RuntimeError("已有评测任务在执行")

        db = SessionLocal()
        try:
            profile = kwargs.get("profile_name") or model_registry.active_profile_name()
            placeholder = EvalRun(profile_name=profile, status="queued")
            db.add(placeholder)
            db.commit()
            db.refresh(placeholder)
            pid = placeholder.id
        finally:
            db.close()

        def _job_wrap() -> None:
            db2 = SessionLocal()
            try:
                old = db2.get(EvalRun, pid)
                if old and old.status == "queued":
                    db2.delete(old)
                    db2.commit()
                self.run_sync(db2, **kwargs)
            except Exception:
                db3 = SessionLocal()
                try:
                    row = db3.get(EvalRun, pid)
                    if row:
                        row.status = "error"
                        db3.commit()
                finally:
                    db3.close()
            finally:
                db2.close()

        threading.Thread(target=_job_wrap, daemon=True, name="eval-runner").start()
        return pid


eval_service = EvalService()
