from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FrozenDict(dict):
    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen mapping does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    RUN_NOT_FOUND = "run_not_found"
    RUN_CONFLICT = "run_conflict"
    MODEL_TIMEOUT = "model_timeout"
    RUN_TIMEOUT = "run_timeout"
    MODEL_RESPONSE_INVALID = "model_response_invalid"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_BUSY = "inference_busy"
    MEMORY_LIMIT = "memory_limit"
    INTERNAL_ERROR = "internal_error"


class ContractError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    document_id: str | None = None
    document_version: str | None = None
    block_id: str | None = None
    text: str = ""
    source_name: str | None = None
    page_no: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    structure_path: str | None = None
    bbox: list[float] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    structured_score: float | None = None
    fusion_rank: int | None = Field(default=None, ge=1)
    fusion_score: float | None = None
    rerank_score: float | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    profile_name: str
    provider: str
    requested_model: str | None = None
    actual_model: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    reranker_model: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    parser_config: dict[str, Any] = Field(default_factory=dict)
    evaluation_run_id: int | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def freeze_nested_configuration(self) -> ModelSnapshot:
        for field_name in (
            "generation_parameters",
            "prompt_versions",
            "retrieval_config",
            "parser_config",
        ):
            object.__setattr__(
                self,
                field_name,
                _freeze_json(getattr(self, field_name)),
            )
        return self


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = None
    thinking: str | None = None
    model_snapshot: ModelSnapshot | None = None
    knowledge_snapshot_id: str | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    latency_sec: float | None = Field(default=None, ge=0)
    memory: dict[str, Any] = Field(default_factory=dict)
    # ---- Agent 编排可观测（阶段 C）----
    # agent_mode=graph 时才有内容；legacy 模式下 trace 为空。
    agent_mode: str | None = None
    route: str | None = None
    retry_retrieve: int = Field(default=0, ge=0)
    retry_generate: int = Field(default=0, ge=0)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    verified: bool = False

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> RunResult:
        if self.status == RunStatus.COMPLETED and self.answer is None:
            raise ValueError("completed run requires answer")
        if self.status == RunStatus.FAILED and self.error_code is None:
            raise ValueError("failed run requires error_code")
        return self


@dataclass(frozen=True)
class KnowledgeSnapshotRef:
    snapshot_id: str
    manifest_path: str


def evidence_from_legacy(raw: dict[str, Any], position: int) -> Evidence:
    source_name = _optional_text(raw.get("source_doc") or raw.get("source_name"))
    block_id = _optional_text(raw.get("block_id"))
    text = str(raw.get("text") or raw.get("content") or "")
    page_no = _positive_int(raw.get("page_no", raw.get("page")))
    structure_path = _optional_text(
        raw.get("structure_path") or raw.get("section_path")
    )
    document_id = (None if raw.get("source_url") else
                   _optional_text(raw.get("document_id") or source_name))
    document_version = _optional_text(raw.get("document_version"))
    identity = {
        "source_doc": source_name,
        "block_id": block_id,
        "page": page_no,
        "text": text,
        "position": int(position),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    quality_flags: list[str] = []
    if not block_id:
        quality_flags.append("missing_block_id")
    if page_no is None and not structure_path:
        quality_flags.append("missing_location")

    known = {
        "text",
        "content",
        "source_doc",
        "source_name",
        "page",
        "page_no",
        "block_id",
        "document_id",
        "document_version",
        "structure_path",
        "section_path",
        "score",
        "dense_score",
        "sparse_score",
        "structured_score",
        "fusion_rank",
        "fusion_score",
        "rerank_score",
        "retrieval_channels",
    }
    metadata: dict[str, Any] = {}
    if raw.get("score") is not None:
        metadata["legacy_score"] = raw["score"]
    if raw.get("source_url"):
        metadata["source_url"] = raw["source_url"]
    extra = {key: value for key, value in raw.items() if key not in known}
    if extra:
        metadata["legacy"] = extra

    return Evidence(
        evidence_id=f"legacy:{digest}",
        document_id=document_id,
        document_version=document_version,
        block_id=block_id,
        text=text,
        source_name=source_name,
        page_no=page_no,
        page_end=_positive_int(raw.get("page_end")),
        structure_path=structure_path,
        dense_score=_optional_float(raw.get("dense_score")),
        sparse_score=_optional_float(raw.get("sparse_score")),
        structured_score=_optional_float(raw.get("structured_score")),
        fusion_rank=_positive_int(raw.get("fusion_rank")),
        fusion_score=_optional_float(raw.get("fusion_score")),
        rerank_score=_optional_float(raw.get("rerank_score")),
        retrieval_channels=[
            str(channel) for channel in (raw.get("retrieval_channels") or [])
        ],
        quality_flags=quality_flags,
        metadata=metadata,
    )


def evidence_from_v2(raw: dict[str, Any], snapshot_id: str) -> Evidence:
    document_version = _optional_text(raw.get("document_version"))
    block_id = _optional_text(raw.get("block_id"))
    if not document_version or not block_id:
        raise ValueError("v2 evidence requires document_version and block_id")
    identity = {
        "snapshot_id": snapshot_id,
        "document_version": document_version,
        "block_id": block_id,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return Evidence(
        evidence_id=f"ev_{digest[:32]}",
        document_id=_optional_text(raw.get("document_id")),
        document_version=document_version,
        block_id=block_id,
        text=str(raw.get("content") or raw.get("text") or ""),
        source_name=_optional_text(raw.get("source_name") or raw.get("source_doc")),
        page_no=_positive_int(raw.get("page") or raw.get("page_no")),
        page_end=_positive_int(raw.get("page_end")),
        structure_path=_optional_text(
            raw.get("structure_path") or raw.get("section_path")
        ),
        bbox=_float_list(raw.get("bbox")),
        dense_score=_optional_float(raw.get("dense_score")),
        sparse_score=_optional_float(raw.get("sparse_score")),
        structured_score=_optional_float(raw.get("structured_score")),
        fusion_rank=_positive_int(raw.get("fusion_rank")),
        fusion_score=_optional_float(raw.get("fusion_score")),
        rerank_score=_optional_float(raw.get("rerank_score")),
        retrieval_channels=[
            str(channel) for channel in (raw.get("retrieval_channels") or [])
        ],
        quality_flags=[str(flag) for flag in (raw.get("quality_flags") or [])],
        metadata={
            "snapshot_id": snapshot_id,
            "chunk_id": raw.get("chunk_id"),
            "source_url": raw.get("source_url"),
            "block_ids": list(raw.get("block_ids") or []),
            "image_score": raw.get("image_score"),
            "image_ref": raw.get("image_ref"),
            "vision_page": raw.get("vision_page"),
            "evidence_kind": raw.get("evidence_kind", "text"),
            "page_id": raw.get("page_id"),
            "image_sha256": raw.get("image_sha256"),
            "graph_score": raw.get("graph_score"),
            "kg_entities": raw.get("kg_entities") or [],
            "kg_relations": raw.get("kg_relations") or [],
            "structural_locator": raw.get("structural_locator"),
            "parent_id": raw.get("parent_id"),
            "block_type": raw.get("type") or raw.get("block_type"),
        },
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value
