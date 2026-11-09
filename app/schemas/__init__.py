# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field

from src.contracts import ErrorCode, Evidence, ModelSnapshot, RunStatus


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)


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
    evidence: List[dict[str, Any]] = Field(default_factory=list)
    latency_sec: Optional[float] = None
    model_profile: Optional[str] = None
    created_at: Optional[datetime] = None


class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[int] = None
    show_thinking: bool = False
    top_k: int = Field(12, ge=1, le=30)
    use_paid_model: bool = True
    use_web_search: bool = True


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageOut
    assistant_message: MessageOut
    elapsed_sec: float
    memory: dict[str, Any] = Field(default_factory=dict)
    profile: Optional[str] = None
    run_id: Optional[str] = None


class RunCreateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[int] = None
    top_k: int = Field(12, ge=1, le=30)
    idempotency_key: Optional[str] = Field(None, min_length=8, max_length=128)
    show_thinking: bool = False
    use_paid_model: bool = True
    use_web_search: bool = True


class ReviewApprovalRequest(BaseModel):
    decision: Literal["accept", "decline"]


class ReviewApprovalOut(BaseModel):
    run_id: str
    decision: Literal["accept", "decline"]
    ticket_id: Optional[str] = None
    ticket_status: Optional[str] = None


class RunOut(BaseModel):
    run_id: str
    status: RunStatus
    answer: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: Optional[float] = None
    thinking: Optional[str] = None
    model_snapshot: Optional[ModelSnapshot] = None
    knowledge_snapshot_id: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None
    latency_sec: Optional[float] = None
    memory: dict[str, Any] = Field(default_factory=dict)
    # ---- Agent 编排可观测（阶段 C）----
    agent_mode: Optional[str] = None
    route: Optional[str] = None
    retry_retrieve: int = 0
    retry_generate: int = 0
    trace: List[dict[str, Any]] = Field(default_factory=list)
    verified: bool = False


class RunTraceOut(BaseModel):
    """Agent 决策轨迹：路由 → 各节点耗时与决策 → 重试计数。"""

    run_id: str
    agent_mode: Optional[str] = None
    route: Optional[str] = None
    retry_retrieve: int = 0
    retry_generate: int = 0
    trace: List[dict[str, Any]] = Field(default_factory=list)
    verified: bool = False
    note: Optional[str] = None


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False


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


class DocumentVersionOut(BaseModel):
    id: str
    status: str
    source_sha256: str
    parser_name: str
    parser_version: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    document_id: str
    display_name: str
    status: str
    active_version_id: Optional[str] = None
    versions: List[DocumentVersionOut] = Field(default_factory=list)


class DocumentIngestResult(BaseModel):
    document_id: str
    display_name: str
    status: str
    active_version_id: Optional[str] = None
    created_version_id: str


class EvidenceDetailOut(BaseModel):
    evidence_kind: str = 'text'
    source_url: Optional[str] = None
    vision_page: int | None = None
    image_available: bool = False
    evidence_id: str
    document_id: Optional[str] = None
    document_version: Optional[str] = None
    block_id: Optional[str] = None
    source_name: Optional[str] = None
    page_no: Optional[int] = None
    page_end: Optional[int] = None
    bbox: Optional[List[float]] = None
    section_path: Optional[str] = None
    structural_locator: Optional[str] = None
    text: str = ""
    table_headers: List[str] = Field(default_factory=list)
    table_rows: List[List[Any]] = Field(default_factory=list)
    unit: Optional[str] = None
    quality_flags: List[str] = Field(default_factory=list)
