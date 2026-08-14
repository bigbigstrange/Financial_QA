# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models import Conversation, Feedback, Message, User
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    FeedbackOut,
    FeedbackRequest,
    MessageOut,
)
from app.services.memory_guard import memory_guard
from app.services.qa_runtime import qa_runtime

router = APIRouter(tags=["chat"])


def _msg_out(m: Message) -> MessageOut:
    evidence = []
    if m.evidence_json:
        try:
            evidence = json.loads(m.evidence_json)
        except json.JSONDecodeError:
            evidence = []
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        thinking=m.thinking,
        confidence=m.confidence,
        evidence=evidence,
        latency_sec=m.latency_sec,
        model_profile=m.model_profile,
        created_at=m.created_at,
    )


@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
        .all()
    )


@router.post("/conversations", response_model=ConversationOut)
def create_conversation(
    body: ConversationCreate,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Conversation:
    conv = Conversation(user_id=user.id, title=body.title[:200])
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/conversations/{conv_id}", response_model=ConversationDetail)
def get_conversation(
    conv_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ConversationDetail:
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[_msg_out(m) for m in conv.messages],
    )


@router.delete("/conversations/{conv_id}")
def delete_conversation(
    conv_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(conv)
    db.commit()
    return {"ok": True}


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问题不能为空")

    if body.conversation_id:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == body.conversation_id, Conversation.user_id == user.id)
            .first()
        )
        if not conv:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        title = q[:40] + ("…" if len(q) > 40 else "")
        conv = Conversation(user_id=user.id, title=title)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    user_msg = Message(conversation_id=conv.id, role="user", content=q)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    qid = f"u{user.id}-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    try:
        result = qa_runtime.answer(q, qid)
    except MemoryError as e:
        memory_guard.persist_last_event(db)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    elapsed = round(time.time() - t0, 2)
    evidence = result.get("evidence") or []
    slim = []
    for ev in evidence[:8]:
        if isinstance(ev, dict):
            slim.append(
                {
                    "text": (ev.get("text") or "")[:400],
                    "source_doc": ev.get("source_doc"),
                    "page": ev.get("page"),
                    "block_id": ev.get("block_id"),
                    "score": ev.get("score"),
                }
            )

    assistant = Message(
        conversation_id=conv.id,
        role="assistant",
        content=result.get("answer") or "",
        thinking=(result.get("thinking") or "") if body.show_thinking else None,
        confidence=float(result.get("confidence") or 0.0),
        evidence_json=json.dumps(slim, ensure_ascii=False),
        latency_sec=elapsed,
        model_profile=result.get("_profile"),
    )
    db.add(assistant)
    conv.title = conv.title or (q[:40] + ("…" if len(q) > 40 else ""))
    db.commit()
    db.refresh(assistant)
    memory_guard.persist_last_event(db)

    return ChatResponse(
        conversation_id=conv.id,
        user_message=_msg_out(user_msg),
        assistant_message=_msg_out(assistant),
        elapsed_sec=elapsed,
        memory=result.get("_memory") or memory_guard.snapshot().to_dict(),
        profile=result.get("_profile"),
    )


@router.post("/feedback", response_model=FeedbackOut)
def feedback(
    body: FeedbackRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Feedback:
    msg = db.query(Message).filter(Message.id == body.message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="消息不存在")
    conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=403, detail="无权反馈该消息")
    row = Feedback(user_id=user.id, message_id=body.message_id, rating=body.rating, comment=body.comment)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
