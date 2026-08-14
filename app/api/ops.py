# -*- coding: utf-8 -*-
"""运维 / 模型迭代 / 内存 / 评测 API（管理员）。"""
from __future__ import annotations

from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.db.session import SessionLocal, get_db
from app.models import EvalRun, MemoryEvent, User
from app.schemas import EvalRunOut, EvalStartRequest, ProfileUpsert, PromoteRequest
from app.services.eval_service import eval_service
from app.services.memory_guard import memory_guard
from app.services.model_registry import model_registry
from app.services.qa_runtime import qa_runtime

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/health")
def health() -> dict[str, Any]:
    """公开健康检查（与 /api/health 一致），便于探针与前端轮询。"""
    st = qa_runtime.status()
    return {"ok": True, **st}


@router.post("/engine/load")
def load_engine(admin: Annotated[User, Depends(require_admin)], profile: Optional[str] = None) -> dict[str, Any]:
    try:
        return qa_runtime.reload(profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/engine/unload-generator")
def unload_generator(admin: Annotated[User, Depends(require_admin)]) -> dict[str, Any]:
    return qa_runtime.unload_generator()


@router.get("/memory")
def memory_status(admin: Annotated[User, Depends(require_admin)]) -> dict[str, Any]:
    return {
        "snapshot": memory_guard.snapshot().to_dict(),
        "last_event": memory_guard.last_event,
    }


@router.get("/memory/events")
def memory_events(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    rows = db.query(MemoryEvent).order_by(MemoryEvent.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "level": r.level,
            "action": r.action,
            "rss_mb": r.rss_mb,
            "pct": r.pct,
            "detail": r.detail,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/models/profiles")
def list_profiles(admin: Annotated[User, Depends(require_admin)]) -> list[dict[str, Any]]:
    return model_registry.list_profiles()


@router.put("/models/profiles/{name}")
def upsert_profile(
    name: str,
    body: ProfileUpsert,
    admin: Annotated[User, Depends(require_admin)],
) -> dict[str, Any]:
    return model_registry.upsert_profile(name, body.model_dump(exclude_unset=True))


@router.post("/models/promote")
def promote(body: PromoteRequest, admin: Annotated[User, Depends(require_admin)]) -> dict[str, Any]:
    try:
        prof = model_registry.promote(body.profile_name, eval_run_id=body.eval_run_id, force=body.force)
        qa_runtime.reload(body.profile_name)
        return {"ok": True, "profile": prof}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/models/rollback")
def rollback(
    admin: Annotated[User, Depends(require_admin)],
    to: Optional[str] = None,
) -> dict[str, Any]:
    try:
        prof = model_registry.rollback(to)
        qa_runtime.reload(prof["name"])
        return {"ok": True, "profile": prof}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/models/gates")
def gates(admin: Annotated[User, Depends(require_admin)]) -> dict[str, Any]:
    return model_registry.gates()


@router.post("/eval/run", response_model=EvalRunOut)
def start_eval(
    body: EvalStartRequest,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> EvalRun:
    try:
        if body.async_mode:
            run_id = eval_service.start_async(
                SessionLocal,
                profile_name=body.profile_name,
                max_samples=body.max_samples,
                promote_if_pass=body.promote_if_pass,
            )
            # 异步刚启动时可能已被内部删除重建；返回最近一条
            run = db.query(EvalRun).order_by(EvalRun.id.desc()).first()
            if not run:
                raise HTTPException(status_code=500, detail="评测启动失败")
            return run
        return eval_service.run_sync(
            db,
            profile_name=body.profile_name,
            max_samples=body.max_samples,
            promote_if_pass=body.promote_if_pass,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/eval/runs", response_model=List[EvalRunOut])
def list_evals(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(30, ge=1, le=100),
) -> list[EvalRun]:
    return db.query(EvalRun).order_by(EvalRun.id.desc()).limit(limit).all()


@router.get("/eval/runs/{run_id}", response_model=EvalRunOut)
def get_eval(
    run_id: int,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> EvalRun:
    run = db.get(EvalRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return run
