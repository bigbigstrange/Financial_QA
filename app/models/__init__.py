# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), index=True, nullable=False)
    rating: Mapped[str] = mapped_column(String(16), nullable=False)  # up | down
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalRun(Base):
    """持续评测运行记录：模型迭代门禁与趋势对比。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_latency_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    gate_passed: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MemoryEvent(Base):
    """内存水位事件：告警 / 拒绝 / 卸载，便于上线后复盘。"""

    __tablename__ = "memory_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(16), nullable=False)  # info|warn|soft|hard
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    rss_mb: Mapped[float] = mapped_column(Float, nullable=False)
    pct: Mapped[float] = mapped_column(Float, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
