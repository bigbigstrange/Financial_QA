from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Annotated, Iterator

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_admin
from app.db.session import SessionLocal, get_db
from app.models import ReviewTicketRecord, RunEventRecord, User
from app.schemas import (ReviewApprovalOut, ReviewApprovalRequest, RunCreateRequest,
                         RunOut, RunTraceOut)
from app.services.run_service import run_service
from src.contracts import ContractError, ErrorCode, RunResult, RunStatus


router = APIRouter(prefix="/runs", tags=["runs"])


def _sse(event: str, data: dict, event_id: int | None = None) -> str:
    identity = f"id: {event_id}\n" if event_id is not None else ""
    return f"{identity}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _safe_pilot_stop_reason() -> str | None:
    path = Path(os.environ.get("FINANCE_PILOT_BUDGET_PATH") or
                Path(__file__).resolve().parents[2] / "output" / "pilot_budget.sqlite3")
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1) as db:
            row = db.execute("SELECT stopped_reason FROM state WHERE id=1").fetchone()
        reason = row[0] if row else None
    except (sqlite3.Error, OSError):
        return None
    return reason if reason in {
        "budget_exhausted", "actual_over_budget", "usage_missing", "usage_invalid",
        "model_mismatch", "http_402", "http_429", "request_uncertain",
    } else None


def run_event_iter(body: RunCreateRequest, user_id: int) -> Iterator[str]:
    """Run in a worker-owned DB session; emit only redacted progress metadata."""
    events: queue.Queue[tuple[str, dict, int | None]] = queue.Queue(maxsize=128)
    closed = threading.Event()
    current_run_id: str | None = None

    def publish(event: str, data: dict, event_id: int | None = None) -> None:
        nonlocal current_run_id
        if event == "started":
            current_run_id = data["run_id"]
        if closed.is_set():
            return
        try:
            events.put_nowait((event, data, event_id))
        except queue.Full:
            # Prefer the terminal marker; dropped progress is replayable from DB.
            if event in {"done", "error", "cancelled"}:
                events.get_nowait()
                events.put_nowait((event, data, event_id))

    def worker() -> None:
        try:
            with SessionLocal() as db:
                def durable_id(run_id: str, event: str) -> int | None:
                    # Callback runs after RunService committed the event in
                    # this worker's session. Never race another connection
                    # against SQLite just to assign an SSE replay cursor.
                    if not hasattr(db, "query"):
                        return None
                    row = (db.query(RunEventRecord.id)
                        .filter(RunEventRecord.run_id == run_id,
                                RunEventRecord.event == event)
                        .order_by(RunEventRecord.id.desc()).first())
                    return row[0] if row else None

                def started(run_id: str) -> None:
                    publish("started", {"run_id": run_id}, durable_id(run_id, "started"))

                def node(trace: dict) -> None:
                    run_id = current_run_id
                    data = {
                        "node": str(trace.get("node") or "")[:60],
                        "decision": str(trace.get("decision") or "")[:120],
                        "elapsed_ms": int(trace.get("elapsed_ms") or 0),
                    }
                    publish("node", data, durable_id(run_id, "node") if run_id else None)

                result = run_service.create_and_execute(
                    db, user_id=user_id, request=body,
                    on_started=started, on_node=node,
                )
                if result.status == RunStatus.FAILED:
                    code = result.error_code.value if result.error_code else ErrorCode.INTERNAL_ERROR.value
                    reason = (_safe_pilot_stop_reason() if code in {
                        ErrorCode.MODEL_UNAVAILABLE.value, ErrorCode.MODEL_RESPONSE_INVALID.value,
                    } else None)
                    publish("error", {"run_id": result.run_id, "code": code,
                                      "reason": reason, "verified": False},
                            durable_id(result.run_id, "error"))
                elif result.status == RunStatus.CANCELLED:
                    publish("cancelled", {"run_id": result.run_id},
                            durable_id(result.run_id, "cancelled"))
                else:
                    publish("done", {"run_id": result.run_id,
                                     "verified": bool(getattr(result, "verified", False))},
                            durable_id(result.run_id, "done"))
        except Exception:
            publish("error", {"code": ErrorCode.INTERNAL_ERROR.value, "verified": False})

    threading.Thread(target=worker, daemon=True, name="run-sse-worker").start()
    try:
        while True:
            event, data, event_id = events.get()
            yield _sse(event, data, event_id)
            if event in {"done", "error", "cancelled"}:
                break
    finally:
        closed.set()


@router.post("/stream")
def stream_run(
    body: RunCreateRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    return StreamingResponse(run_event_iter(body, user.id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{run_id}/events")
def replay_run_events(
    run_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    if run_service.get_owned(db, run_id=run_id, user_id=user.id) is None:
        raise HTTPException(status_code=404, detail={"code": ErrorCode.RUN_NOT_FOUND.value,
            "message": "运行不存在"})
    try:
        after_id = int(last_event_id or "0")
        if after_id < 0:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": ErrorCode.INVALID_REQUEST.value,
            "message": "Last-Event-ID 必须是非负整数"}) from exc

    def stream() -> Iterator[str]:
        cursor = after_id
        while True:
            with SessionLocal() as session:
                events = run_service.list_events(session, run_id=run_id,
                    user_id=user.id, after_id=cursor)
                if events is None:
                    return
                frames = [(item.id, _sse(item.event, json.loads(item.data_json), item.id))
                          for item in events]
                record = run_service.get_owned(session, run_id=run_id, user_id=user.id)
                terminal = record.status in {RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value, RunStatus.CANCELLED.value}
            for event_id, frame in frames:
                cursor = event_id
                yield frame
            if terminal:
                return
            time.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/{run_id}/cancel", response_model=RunOut)
def cancel_run(
    run_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> RunOut:
    result = run_service.cancel(db, run_id=run_id, user_id=user.id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": ErrorCode.RUN_NOT_FOUND.value,
            "message": "运行不存在"})
    return RunOut.model_validate(result.model_dump())


@router.post("/{run_id}/approvals", response_model=ReviewApprovalOut)
def decide_review(
    run_id: str,
    body: ReviewApprovalRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ReviewApprovalOut:
    try:
        result = run_service.decide_review(db, run_id=run_id, user_id=user.id,
                                           decision=body.decision)
    except ContractError as exc:
        raise HTTPException(status_code=_status_for_error(exc.code),
            detail={"code": exc.code.value, "message": str(exc)}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail={"code": ErrorCode.RUN_NOT_FOUND.value,
            "message": "运行不存在"})
    return ReviewApprovalOut.model_validate(result)


@router.post("", response_model=RunOut)
def create_run(
    body: RunCreateRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> RunOut:
    try:
        result = run_service.create_and_execute(
            db,
            user_id=user.id,
            request=body,
        )
    except ContractError as exc:
        raise HTTPException(
            status_code=_status_for_error(exc.code),
            detail={"code": exc.code.value, "message": str(exc)},
        ) from exc
    raise_for_failed_result(result)
    return RunOut.model_validate(result.model_dump())


@router.get("/review-tickets")
def list_review_tickets(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    query = db.query(ReviewTicketRecord)
    total = query.count()
    rows = (query.order_by(ReviewTicketRecord.created_at.desc(),
                           ReviewTicketRecord.id.desc())
            .offset(offset).limit(limit).all())
    return {"total": total, "limit": limit, "offset": offset,
        "items": [{"id": row.id, "run_id": row.run_id,
                   "user_id": row.user_id, "reason": row.reason,
                   "status": row.status,
                   "created_at": row.created_at.isoformat() if row.created_at else None}
                  for row in rows]}


@router.get("/by-key/{idempotency_key}", response_model=RunOut)
def get_run_by_key(
    idempotency_key: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> RunOut:
    record = run_service.get_owned_by_key(db, key=idempotency_key, user_id=user.id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": ErrorCode.RUN_NOT_FOUND.value,
            "message": "运行不存在"})
    return RunOut.model_validate(run_service.to_result(record).model_dump())


@router.get("/{run_id}", response_model=RunOut)
def get_run(
    run_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> RunOut:
    record = run_service.get_owned(db, run_id=run_id, user_id=user.id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.RUN_NOT_FOUND.value,
                "message": "运行不存在",
            },
        )
    result = run_service.to_result(record)
    return RunOut.model_validate(result.model_dump())


@router.get("/{run_id}/trace", response_model=RunTraceOut)
def get_run_trace(
    run_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> RunTraceOut:
    """返回该 run 的 Agent 决策轨迹（路由、各节点耗时与决策、重试计数）。

    仅 `FINANCE_AGENT_MODE=graph` 时才有轨迹；legacy 运行会返回空轨迹并附说明，
    而不是 404 —— 便于前端统一渲染。
    """
    record = run_service.get_owned(db, run_id=run_id, user_id=user.id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.RUN_NOT_FOUND.value,
                "message": "运行不存在",
            },
        )
    return RunTraceOut.model_validate(run_service.to_trace(record))


def raise_for_failed_result(result: RunResult) -> None:
    if result.status != RunStatus.FAILED:
        return
    code = result.error_code or ErrorCode.INTERNAL_ERROR
    raise HTTPException(
        status_code=_status_for_error(code),
        detail={
            "code": code.value,
            "message": result.error_message or "运行失败",
            "run_id": result.run_id,
        },
    )


def _status_for_error(code: ErrorCode) -> int:
    if code == ErrorCode.INVALID_REQUEST:
        return 400
    if code == ErrorCode.RUN_NOT_FOUND:
        return 404
    if code == ErrorCode.RUN_CONFLICT:
        return 409
    if code == ErrorCode.INFERENCE_BUSY:
        return 429
    if code in {
        ErrorCode.MEMORY_LIMIT,
        ErrorCode.MODEL_TIMEOUT,
        ErrorCode.MODEL_UNAVAILABLE,
    }:
        return 503
    return 500
