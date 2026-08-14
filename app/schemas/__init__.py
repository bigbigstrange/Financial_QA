# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    username: str
    # 用 str：避免 .local 等开发域名被 email-validator 拒绝导致 /auth/me 500
    email: str
    role: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ConversationCreate(BaseModel):
    title: str = Field("新会话", max_length=200)


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    id: int
    role: str
    content: str
    thinking: Optional[str] = None
    confidence: Optional[float] = None
    evidence: List[dict[str, Any]] = []
    latency_sec: Optional[float] = None
    model_profile: Optional[str] = None
    created_at: Optional[datetime] = None


class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = []


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[int] = None
    show_thinking: bool = False
    top_k: int = Field(12, ge=1, le=30)


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageOut
    assistant_message: MessageOut
    elapsed_sec: float
    memory: dict[str, Any] = {}
    profile: Optional[str] = None


class FeedbackRequest(BaseModel):
    message_id: int
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None


class FeedbackOut(BaseModel):
    id: int
    message_id: int
    rating: str
    comment: Optional[str] = None

    model_config = {"from_attributes": True}


class ProfileUpsert(BaseModel):
    label: Optional[str] = None
    embed_model: Optional[str] = None
    reranker_model: Optional[str] = None
    gen_model: Optional[str] = None
    image_backend: Optional[str] = None
    enable_graph: Optional[bool] = None
    notes: Optional[str] = None


class PromoteRequest(BaseModel):
    profile_name: str
    eval_run_id: Optional[int] = None
    force: bool = False


class EvalStartRequest(BaseModel):
    profile_name: Optional[str] = None
    max_samples: int = Field(20, ge=1, le=500)
    promote_if_pass: bool = False
    async_mode: bool = True


class EvalRunOut(BaseModel):
    id: int
    profile_name: str
    status: str
    samples: int
    success_rate: Optional[float] = None
    avg_latency_sec: Optional[float] = None
    avg_confidence: Optional[float] = None
    evidence_coverage: Optional[float] = None
    gate_passed: int
    report_path: Optional[str] = None
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
