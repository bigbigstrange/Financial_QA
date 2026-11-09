from __future__ import annotations

import json
import hashlib
import logging
import os
import time
import uuid
from datetime import timezone
from typing import Any, Callable

from sqlalchemy.exc import IntegrityError
from sqlalchemy import MetaData, Table, update
from sqlalchemy.orm import Session

from app.models import (
    Conversation,
    EvidenceRecord,
    KnowledgeSnapshotRecord,
    RunRecord,
    RunEventRecord,
    ReviewTicketRecord,
    ReviewApprovalRecord,
)
from app.schemas import RunCreateRequest
from app.services.model_registry import model_registry
from app.services.qa_runtime import QARuntime, qa_runtime
from src.system.cost_policy import paid_api_disabled
from src.contracts import (
    ContractError,
    ErrorCode,
    Evidence,
    KnowledgeSnapshotRef,
    ModelSnapshot,
    RunResult,
    RunStatus,
    evidence_from_legacy,
    evidence_from_v2,
    utc_now,
)


class RunCancelled(Exception):
    pass


class RunService:
    def __init__(
        self,
        *,
        qa_runtime: Any,
        registry: Any,
        knowledge_registry: Any = None,
    ) -> None:
        self.qa_runtime = qa_runtime
        self.registry = registry
        self.knowledge_registry = knowledge_registry

    def create_and_execute(
        self,
        db: Session,
        *,
        user_id: int,
        request: RunCreateRequest,
        on_started: Callable[[str], None] | None = None,
        on_node: Callable[[dict], None] | None = None,
    ) -> RunResult:
        question = request.question.strip()
        if not question:
            raise ContractError(ErrorCode.INVALID_REQUEST, "问题不能为空")
        options = {"use_paid_model": request.use_paid_model,
                   "use_web_search": request.use_web_search}

        if request.idempotency_key:
            existing = (
                db.query(RunRecord)
                .filter(
                    RunRecord.user_id == user_id,
                    RunRecord.idempotency_key == request.idempotency_key,
                )
                .first()
            )
            if existing is not None:
                if existing.question != question or self._load_options(existing) != options:
                    raise ContractError(
                        ErrorCode.RUN_CONFLICT,
                        "幂等键已用于其他问题或问答开关",
                    )
                return self.to_result(
                    existing,
                    include_thinking=request.show_thinking,
                )

        if request.conversation_id is not None:
            conversation = (
                db.query(Conversation)
                .filter(
                    Conversation.id == request.conversation_id,
                    Conversation.user_id == user_id,
                )
                .first()
            )
            if conversation is None:
                raise ContractError(
                    ErrorCode.RUN_NOT_FOUND,
                    "会话不存在",
                )

        snapshot = self.registry.snapshot()
        if paid_api_disabled() or not request.use_paid_model:
            snapshot = ModelSnapshot(snapshot_id="runtime:local", profile_name="local-retrieval",
                                     provider="local", requested_model=None, actual_model=None)
        elif (os.environ.get("FINANCE_LLM_PROVIDER") or "").strip().lower() == "ark":
            snapshot = ModelSnapshot(
                snapshot_id="runtime:ark:requested", profile_name="runtime",
                provider="ark",
                requested_model=(os.environ.get("ARK_MODEL") or
                                 "doubao-seed-2-1-pro-260915").strip(),
                actual_model=None,
            )
        knowledge = self._knowledge_ref(db)
        run_id = str(uuid.uuid4())
        record = RunRecord(
            id=run_id,
            user_id=user_id,
            conversation_id=request.conversation_id,
            idempotency_key=request.idempotency_key,
            options_json=json.dumps(options, sort_keys=True),
            status=RunStatus.PENDING.value,
            question=question,
            model_snapshot_json=snapshot.model_dump_json(),
            knowledge_snapshot_id=(
                knowledge.snapshot_id if knowledge is not None else None
            ),
        )
        db.add(record)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            existing = None
            if request.idempotency_key:
                existing = (
                    db.query(RunRecord)
                    .filter(
                        RunRecord.user_id == user_id,
                        RunRecord.idempotency_key == request.idempotency_key,
                    )
                    .first()
                )
            if existing is None:
                raise ContractError(
                    ErrorCode.RUN_CONFLICT,
                    "运行创建冲突",
                ) from exc
            if existing.question != question or self._load_options(existing) != options:
                raise ContractError(
                    ErrorCode.RUN_CONFLICT,
                    "幂等键已用于其他问题或问答开关",
                ) from exc
            return self.to_result(
                existing,
                include_thinking=request.show_thinking,
            )
        except Exception as exc:
            db.rollback()
            raise ContractError(
                ErrorCode.INTERNAL_ERROR,
                "无法创建运行记录",
            ) from exc

        inference_active = False
        started = time.perf_counter()
        try:
            self._event(db, record, "started", {"run_id": run_id})
            db.commit()
            if on_started is not None:
                on_started(run_id)
            transitioned = db.execute(update(RunRecord)
                .where(RunRecord.id == run_id,
                       RunRecord.status == RunStatus.PENDING.value,
                       RunRecord.cancel_requested_at.is_(None))
                .values(status=RunStatus.RUNNING.value, started_at=utc_now())).rowcount
            db.commit()
            db.refresh(record)
            if not transitioned:
                return self.to_result(record, include_thinking=request.show_thinking)

            inference_active = True
            def check_cancel() -> None:
                # LangGraph may call this guard from its worker thread. Give
                # that thread its own session rather than sharing the API session.
                with Session(bind=db.get_bind()) as current_db:
                    current = current_db.get(RunRecord, run_id)
                    if current is None or current.cancel_requested_at is not None or current.status == RunStatus.CANCELLED.value:
                        raise RunCancelled()
                from app.core.config import get_settings
                if time.perf_counter() - started > get_settings().run_timeout_seconds:
                    raise ContractError(ErrorCode.RUN_TIMEOUT, "问答超过总运行时限")

            def checkpoint_node(trace: dict) -> None:
                check_cancel()
                node = {
                    "node": str(trace.get("node") or "")[:60],
                    "decision": str(trace.get("decision") or "")[:120],
                    "elapsed_ms": int(trace.get("elapsed_ms") or 0),
                }
                record.checkpoint_json = json.dumps(node, ensure_ascii=False)
                self._event(db, record, "node", node)
                db.commit()
                if on_node is not None:
                    on_node(node)

            answer_kwargs = {"knowledge_snapshot": knowledge}
            if on_node is not None or isinstance(self.qa_runtime, QARuntime):
                answer_kwargs["on_node"] = checkpoint_node
            if isinstance(self.qa_runtime, QARuntime):
                answer_kwargs["cancel_check"] = check_cancel
                answer_kwargs["use_paid_model"] = request.use_paid_model
                answer_kwargs["use_web_search"] = request.use_web_search
            raw = self.qa_runtime.answer(question, run_id, **answer_kwargs)
            inference_active = False
            check_cancel()
            self._complete_record(db, record, raw, knowledge=knowledge,
                                  show_thinking=request.show_thinking,
                                  latency_sec=time.perf_counter() - started)
            return self.to_result(
                record,
                include_thinking=request.show_thinking,
                memory=raw.get("_memory") or {},
            )
        except Exception as exc:
            db.rollback()
            cancelled = isinstance(exc, RunCancelled)
            code, message = self._map_error(exc, inference_active=inference_active)
            if not cancelled:
                logging.getLogger(__name__).error("Run failed: run_id=%s type=%s",
                                                  run_id, type(exc).__name__)
            failed = db.get(RunRecord, run_id)
            if failed is None:
                return RunResult(
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    model_snapshot=snapshot,
                    error_code=code,
                    error_message=message,
                    latency_sec=time.perf_counter() - started,
                )
            failed.status = (RunStatus.CANCELLED if cancelled else RunStatus.FAILED).value
            failed.error_code = None if cancelled else code.value
            failed.error_message = None if cancelled else message
            failed.latency_sec = time.perf_counter() - started
            failed.finished_at = utc_now()
            if cancelled:
                self._event(db, failed, "cancelled", {"run_id": run_id})
            else:
                self._event(db, failed, "error", {"run_id": run_id,
                    "code": code.value, "verified": False})
                self._review(db, failed, "failed")
            db.commit()
            return self.to_result(
                failed,
                include_thinking=request.show_thinking,
            )

    def _complete_record(self, db: Session, record: RunRecord, raw: dict,
                         *, knowledge: KnowledgeSnapshotRef | None,
                         show_thinking: bool, latency_sec: float) -> None:
        # Acquire the database write lock while the run is still cancellable.
        claimed = db.execute(update(RunRecord)
            .where(RunRecord.id == record.id,
                   RunRecord.status == RunStatus.RUNNING.value,
                   RunRecord.cancel_requested_at.is_(None))
            .values(status=RunStatus.RUNNING.value)).rowcount
        if not claimed:
            raise RunCancelled()
        runtime_snapshot = raw.get("_model_snapshot")
        if isinstance(runtime_snapshot, ModelSnapshot):
            record.model_snapshot_json = runtime_snapshot.model_dump_json()
        elif raw.get("_local_only") is True:
            record.model_snapshot_json = ModelSnapshot(
                snapshot_id="runtime:local", profile_name="local-retrieval",
                provider="local", requested_model=None, actual_model=None,
            ).model_dump_json()
        evidence = self._convert_evidence(raw.get("evidence"),
            snapshot_id=knowledge.snapshot_id if knowledge is not None else None)
        evidence = [self._scope_evidence(item, run_id=record.id, user_id=record.user_id)
                    for item in evidence]
        record.answer = str(raw.get("answer") or "")
        record.evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence], ensure_ascii=False)
        record.confidence = _optional_float(raw.get("confidence"))
        record.agent_trace_json = self._trace_json(raw)
        record.thinking = str(raw.get("thinking") or "") or None if show_thinking else None
        record.latency_sec = latency_sec
        record.status = RunStatus.COMPLETED.value
        record.finished_at = utc_now()
        for item in evidence:
            if db.get(EvidenceRecord, item.evidence_id) is None:
                self._insert_evidence(db, item, run_id=record.id, user_id=record.user_id)
        self._event(db, record, "done", {"run_id": record.id,
            "verified": bool(raw.get("verified"))})
        if not raw.get("verified"):
            self._review(db, record, "uncertain")
        db.commit()

    @staticmethod
    def _event(db: Session, record: RunRecord, event: str, data: dict) -> None:
        db.add(RunEventRecord(run_id=record.id, user_id=record.user_id,
            event=event, data_json=json.dumps(data, ensure_ascii=False)))

    @staticmethod
    def _review(db: Session, record: RunRecord, reason: str) -> None:
        if db.query(ReviewTicketRecord).filter_by(run_id=record.id).first() is None:
            db.add(ReviewTicketRecord(id=str(uuid.uuid4()), run_id=record.id,
                user_id=record.user_id, reason=reason, status="suggested"))

    def decide_review(self, db: Session, *, run_id: str, user_id: int,
                      decision: str) -> dict | None:
        """User-owned, durable one-shot decision; only acceptance opens a ticket."""
        record = self.get_owned(db, run_id=run_id, user_id=user_id)
        if record is None:
            return None
        if record.status not in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
            raise ContractError(ErrorCode.RUN_CONFLICT, "仅已结束的问答可申请知识核查")
        if decision not in {"accept", "decline"}:
            raise ContractError(ErrorCode.INVALID_REQUEST, "无效的确认决定")
        existing = db.query(ReviewApprovalRecord).filter_by(run_id=run_id).first()
        if existing is not None:
            if existing.user_id != user_id or existing.decision != decision:
                raise ContractError(ErrorCode.RUN_CONFLICT, "该问答已有不同的核查决定")
        else:
            db.add(ReviewApprovalRecord(id=str(uuid.uuid4()), run_id=run_id,
                                        user_id=user_id, decision=decision))
            if decision == "accept":
                ticket = db.query(ReviewTicketRecord).filter_by(run_id=run_id).first()
                if ticket is None:
                    ticket = ReviewTicketRecord(id=str(uuid.uuid4()), run_id=run_id,
                        user_id=user_id, reason="user_requested", status="open")
                    db.add(ticket)
                else:
                    ticket.status = "open"
            self._event(db, record, "review_decision", {"run_id": run_id,
                "decision": decision})
            try:
                db.commit()
            except IntegrityError as exc:
                db.rollback()
                existing = db.query(ReviewApprovalRecord).filter_by(run_id=run_id).first()
                if existing is None or existing.user_id != user_id or existing.decision != decision:
                    raise ContractError(ErrorCode.RUN_CONFLICT, "核查决定冲突") from exc
        ticket = db.query(ReviewTicketRecord).filter_by(run_id=run_id).first()
        return {"run_id": run_id, "decision": decision,
                "ticket_id": ticket.id if decision == "accept" and ticket else None,
                "ticket_status": ticket.status if decision == "accept" and ticket else None}

    def list_events(self, db: Session, *, run_id: str, user_id: int,
                    after_id: int = 0) -> list[RunEventRecord] | None:
        if self.get_owned(db, run_id=run_id, user_id=user_id) is None:
            return None
        return (db.query(RunEventRecord)
            .filter(RunEventRecord.run_id == run_id, RunEventRecord.user_id == user_id,
                    RunEventRecord.id > after_id)
            .order_by(RunEventRecord.id).all())

    def cancel(self, db: Session, *, run_id: str, user_id: int) -> RunResult | None:
        record = self.get_owned(db, run_id=run_id, user_id=user_id)
        if record is None:
            return None
        now = utc_now()
        pending = db.execute(update(RunRecord)
            .where(RunRecord.id == run_id, RunRecord.user_id == user_id,
                   RunRecord.status == RunStatus.PENDING.value)
            .values(status=RunStatus.CANCELLED.value,
                    cancel_requested_at=now, finished_at=now)).rowcount
        if pending:
            self._event(db, record, "cancelled", {"run_id": run_id})
            db.commit()
        else:
            db.execute(update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.user_id == user_id,
                       RunRecord.status == RunStatus.RUNNING.value,
                       RunRecord.cancel_requested_at.is_(None))
                .values(cancel_requested_at=now))
            db.commit()
        db.refresh(record)
        return self.to_result(record)

    def recover_interrupted_runs(self, db: Session) -> int:
        """Reconcile orphaned runs without repeating an uncertain model call."""
        records = db.query(RunRecord).filter(RunRecord.status.in_([
            RunStatus.PENDING.value, RunStatus.RUNNING.value])).all()
        for record in records:
            recovery_error = None
            if (record.status == RunStatus.RUNNING.value
                    and record.cancel_requested_at is None
                    and record.checkpoint_json
                    and hasattr(self.qa_runtime, "resume_graph")):
                knowledge = None
                if record.knowledge_snapshot_id:
                    snapshot = db.get(KnowledgeSnapshotRecord, record.knowledge_snapshot_id)
                    if snapshot is not None:
                        knowledge = KnowledgeSnapshotRef(snapshot_id=snapshot.id,
                            manifest_path=snapshot.manifest_path)
                if not record.knowledge_snapshot_id or knowledge is not None:
                    try:
                        from app.core.config import get_settings
                        from src.system.execution_control import execution_guard

                        run_id = record.id
                        origin = record.started_at or record.created_at
                        if origin is not None and origin.tzinfo is None:
                            origin = origin.replace(tzinfo=timezone.utc)

                        def check_recovery():
                            with Session(bind=db.get_bind()) as current_db:
                                current = current_db.get(RunRecord, run_id)
                                if (current is None or current.cancel_requested_at is not None
                                        or current.status != RunStatus.RUNNING.value):
                                    raise RunCancelled()
                            if origin and (utc_now() - origin).total_seconds() > get_settings().run_timeout_seconds:
                                raise ContractError(ErrorCode.RUN_TIMEOUT, "恢复任务已超过总运行时限")

                        guard_token = execution_guard.set(check_recovery)
                        try:
                            check_recovery()
                            raw = self.qa_runtime.resume_graph(record.question, run_id,
                                                               knowledge_snapshot=knowledge)
                            check_recovery()
                        finally:
                            execution_guard.reset(guard_token)
                        db.refresh(record)
                        if raw is not None and record.cancel_requested_at is None:
                            elapsed = (utc_now() - origin).total_seconds() if origin else 0.0
                            self._complete_record(db, record, raw, knowledge=knowledge,
                                                  show_thinking=False,
                                                  latency_sec=max(0.0, elapsed))
                            continue
                    except Exception as exc:
                        db.rollback()
                        recovery_error = exc if isinstance(exc, ContractError) else None
                        logging.getLogger(__name__).error(
                            "Run recovery failed: run_id=%s type=%s",
                            record.id, type(exc).__name__)
                    record = db.get(RunRecord, record.id)
            if record.cancel_requested_at is not None:
                record.status = RunStatus.CANCELLED.value
                self._event(db, record, "cancelled", {"run_id": record.id})
            else:
                record.status = RunStatus.FAILED.value
                record.error_code = (recovery_error.code if recovery_error else ErrorCode.INTERNAL_ERROR).value
                record.error_message = str(recovery_error) if recovery_error else "运行中断，已提交人工复核"
                self._event(db, record, "error", {"run_id": record.id,
                    "code": record.error_code, "verified": False})
                self._review(db, record, "interrupted")
            record.finished_at = utc_now()
        db.commit()
        return len(records)

    @staticmethod
    def _scope_evidence(item, *, run_id, user_id):
        """A source identity is reusable; an owner's recorded hit is not."""
        identity = json.dumps([run_id, user_id, item.evidence_id, item.metadata.get('page_id')])
        return item.model_copy(update={
            'evidence_id': 'ev_' + uuid.uuid5(uuid.NAMESPACE_URL, identity).hex,
            'metadata': {**item.metadata, 'source_evidence_id': item.evidence_id},
        })

    @staticmethod
    def _insert_evidence(db, item, *, run_id, user_id):
        # Historical deployments have required legacy columns and a composite
        # primary key. Keep that schema/history intact and populate both shapes.
        table = Table("evidence_records", MetaData(), autoload_with=db.connection())
        source_url = item.metadata.get("source_url")
        if source_url:
            # Historical local-evidence tables require document/version/block
            # fields. Web sources have none; these storage-only IDs are never
            # exposed as local DocIR identities.
            from src.system.web_search import _public_url
            if _public_url(source_url) != source_url:
                raise ValueError("invalid web evidence URL")
            source_key = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:32]
            document_id = f"web:{source_key}"
            document_version = f"web:{source_key}"
            block_id = f"web:{hashlib.sha256(item.evidence_id.encode()).hexdigest()[:32]}"
        else:
            document_id = item.document_id
            document_version = item.document_version
            block_id = item.block_id
        values = dict(id=item.evidence_id, run_id=run_id, user_id=user_id,
            document_id=document_id, document_version_id=(None if source_url else document_version),
            block_id=block_id, evidence_json=item.model_dump_json())
        legacy = dict(snapshot_id=item.metadata.get("snapshot_id") or "",
            document_version=document_version or "", quote=item.text,
            scores_json=json.dumps({"fusion_score": item.fusion_score}),
            structural_locator=item.metadata.get("structural_locator"),
            page=item.page_no, bbox_json=json.dumps(item.bbox))
        values.update({k: v for k, v in legacy.items() if k in table.c})
        db.execute(table.insert().values(**values))

    def get_owned(
        self,
        db: Session,
        *,
        run_id: str,
        user_id: int,
    ) -> RunRecord | None:
        return (
            db.query(RunRecord)
            .filter(RunRecord.id == run_id, RunRecord.user_id == user_id)
            .first()
        )

    @staticmethod
    def get_owned_by_key(db: Session, *, key: str, user_id: int) -> RunRecord | None:
        if len(key) < 8 or len(key) > 128:
            return None
        return (db.query(RunRecord)
            .filter(RunRecord.idempotency_key == key, RunRecord.user_id == user_id)
            .first())

    def to_result(
        self,
        record: RunRecord,
        *,
        include_thinking: bool = False,
        memory: dict[str, Any] | None = None,
    ) -> RunResult:
        evidence_data = json.loads(record.evidence_json or "[]")
        snapshot_data = json.loads(record.model_snapshot_json)
        error_code = ErrorCode(record.error_code) if record.error_code else None
        agent = self._load_agent_payload(record)
        return RunResult(
            run_id=record.id,
            status=RunStatus(record.status),
            answer=record.answer,
            evidence=[Evidence.model_validate(item) for item in evidence_data],
            confidence=record.confidence,
            thinking=record.thinking if include_thinking else None,
            model_snapshot=ModelSnapshot.model_validate(snapshot_data),
            knowledge_snapshot_id=record.knowledge_snapshot_id,
            error_code=error_code,
            error_message=record.error_message,
            latency_sec=record.latency_sec,
            memory=dict(memory or {}),
            agent_mode=agent.get("agent_mode"),
            route=agent.get("route"),
            retry_retrieve=int(agent.get("retry_retrieve") or 0),
            retry_generate=int(agent.get("retry_generate") or 0),
            trace=list(agent.get("trace") or []),
            verified=bool(agent.get("verified")),
        )

    def to_trace(self, record: RunRecord) -> dict[str, Any]:
        """轻量返回 Agent 编排轨迹（不加载证据与模型快照）。

        无论有无轨迹都返回**完整形状**，便于前端与调用方统一处理。
        """
        payload = self._load_agent_payload(record)
        result: dict[str, Any] = {
            "run_id": record.id,
            "agent_mode": payload.get("agent_mode"),
            "route": payload.get("route"),
            "retry_retrieve": int(payload.get("retry_retrieve") or 0),
            "retry_generate": int(payload.get("retry_generate") or 0),
            "trace": list(payload.get("trace") or []),
            "verified": bool(payload.get("verified")),
        }
        if not result["trace"]:
            result["note"] = (
                "该运行没有 Agent 轨迹。仅当 FINANCE_AGENT_MODE=graph 且使用 /api/runs 时才会记录。"
            )
        return result

    @staticmethod
    def _trace_json(raw: dict[str, Any]) -> str | None:
        """把 Agent 编排轨迹序列化落库；legacy 模式没有 trace，返回 None。"""
        trace = raw.get("trace")
        if not trace:
            return None
        return json.dumps(
            {
                "agent_mode": raw.get("agent_mode"),
                "route": raw.get("route"),
                "retry_retrieve": int(raw.get("retry_retrieve") or 0),
                "retry_generate": int(raw.get("retry_generate") or 0),
                "trace": trace,
                "verified": bool(raw.get("verified")),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _load_agent_payload(record: RunRecord) -> dict[str, Any]:
        if not record.agent_trace_json:
            return {}
        try:
            payload = json.loads(record.agent_trace_json)
        except (TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _load_options(record: RunRecord) -> dict[str, bool]:
        try:
            data = json.loads(record.options_json or "{}")
        except (TypeError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        # Runs created before this column existed used both capabilities.
        return {"use_paid_model": data.get("use_paid_model", True) is True,
                "use_web_search": data.get("use_web_search", True) is True}

    @staticmethod
    def _convert_evidence(
        raw_items: Any,
        *,
        snapshot_id: str | None,
    ) -> list[Evidence]:
        if not isinstance(raw_items, list):
            return []
        out: list[Evidence] = []
        for index, item in enumerate(raw_items[:8]):
            if not isinstance(item, dict):
                continue
            if snapshot_id and item.get("document_version") and item.get("block_id"):
                out.append(evidence_from_v2(item, snapshot_id))
            else:
                out.append(evidence_from_legacy(item, position=index))
        return out

    def _knowledge_ref(self, db: Session) -> KnowledgeSnapshotRef | None:
        if self.knowledge_registry is not None:
            return self.knowledge_registry.active_ref(db)
        record = (
            db.query(KnowledgeSnapshotRecord)
            .filter(KnowledgeSnapshotRecord.status == "active")
            .first()
        )
        if record is None:
            return None
        return KnowledgeSnapshotRef(
            snapshot_id=record.id,
            manifest_path=record.manifest_path,
        )

    @staticmethod
    def _map_error(
        exc: Exception,
        *,
        inference_active: bool,
    ) -> tuple[ErrorCode, str]:
        if isinstance(exc, ContractError):
            return exc.code, str(exc)
        if inference_active and isinstance(exc, MemoryError):
            return ErrorCode.MEMORY_LIMIT, str(exc) or "内存资源不足"
        if inference_active and isinstance(exc, RuntimeError):
            return ErrorCode.INFERENCE_BUSY, str(exc) or "推理服务繁忙"
        return ErrorCode.INTERNAL_ERROR, "运行失败，请稍后重试"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


run_service = RunService(qa_runtime=qa_runtime, registry=model_registry)
