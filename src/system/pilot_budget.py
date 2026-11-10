"""Fail-closed, cross-process token ledger for the Ark pilot."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.contracts import ContractError, ErrorCode
from src.models.gateway import ModelResponse


class PilotBudgetStopped(ContractError):
    def __init__(self, message: str, code: ErrorCode = ErrorCode.MODEL_UNAVAILABLE) -> None:
        super().__init__(code, message, retryable=False)


class PilotBudget:
    MAX_APPROVED_TOKENS = 1_500_000

    def __init__(
        self,
        path: Path,
        limit: int = 1_500_000,
        model: str = "doubao-seed-2-1-pro-260915",
    ) -> None:
        self.path = Path(path)
        if int(limit) < 1:
            raise ValueError("试跑预算必须为正数")
        self.limit = min(int(limit), self.MAX_APPROVED_TOKENS)
        self.model = model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                request_id TEXT PRIMARY KEY,
                reserved INTEGER NOT NULL,
                status TEXT NOT NULL,
                node TEXT,
                provider TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms REAL,
                reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(calls)")}
            if "latency_ms" not in columns:
                db.execute("ALTER TABLE calls ADD COLUMN latency_ms REAL")
            db.execute("""CREATE TABLE IF NOT EXISTS state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                stopped_reason TEXT
            )""")
            db.execute("INSERT OR IGNORE INTO state(id, stopped_reason) VALUES(1, NULL)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _totals(db: sqlite3.Connection) -> tuple[int, int, int]:
        spent, reserved, uncertain = db.execute("""SELECT
            COALESCE(SUM(CASE WHEN status='settled' THEN total_tokens ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN status='reserved' THEN reserved ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN status='uncertain' THEN reserved ELSE 0 END), 0)
            FROM calls""").fetchone()
        return int(spent), int(reserved), int(uncertain)

    @staticmethod
    def _stopped(db: sqlite3.Connection) -> bool:
        return db.execute("SELECT stopped_reason FROM state WHERE id=1").fetchone()[0] is not None

    def reserve(self, request_id: str, tokens: int, node: str = "chat") -> None:
        if tokens <= 0:
            raise ValueError("预留 tokens 必须大于零")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if self._stopped(db):
                raise PilotBudgetStopped("试跑已暂停，禁止后续付费调用")
            spent, pending, uncertain = self._totals(db)
            if pending:
                raise PilotBudgetStopped("已有一笔模型调用进行中，试跑仅允许串行请求")
            if spent + pending + uncertain + tokens > self.limit:
                db.execute("UPDATE state SET stopped_reason='budget_exhausted' WHERE id=1")
                db.commit()
                raise PilotBudgetStopped("试跑预算不足，已停止付费调用")
            db.execute(
                "INSERT INTO calls(request_id,reserved,status,node,provider,model) VALUES(?,?,'reserved',?,'ark',?)",
                (request_id, tokens, node, self.model),
            )
            db.commit()

    @staticmethod
    def _same_model(requested: str, actual: str) -> bool:
        """Ark 会返回带版本后缀的模型名（如 -latest-version），按前缀兼容。"""
        if actual == requested:
            return True
        return actual.startswith(requested + "-") or requested.startswith(actual + "-")

    def settle(self, request_id: str, response: ModelResponse, node: str) -> None:
        valid_usage = all(
            isinstance(value, int) and value >= 0
            for value in (response.prompt_tokens, response.completion_tokens, response.total_tokens)
        )
        if valid_usage:
            valid_usage = (response.prompt_tokens > 0 and
                           response.total_tokens >= response.prompt_tokens + response.completion_tokens)
        valid_model = (
            response.provider == "ark"
            and self._same_model(self.model, response.actual_model or "")
        )
        if not valid_usage or not valid_model:
            self.stop(request_id, "usage_invalid" if not valid_usage else "model_mismatch")
            code = ErrorCode.MODEL_RESPONSE_INVALID
            raise PilotBudgetStopped("模型用量缺失、不一致或返回模型不符，试跑已暂停", code)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM calls WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None or row[0] != "reserved":
                raise PilotBudgetStopped("模型调用没有有效预算预留")
            db.execute("""UPDATE calls SET reserved=0,status='settled',node=?,latency_ms=?,
                provider=?,model=?,prompt_tokens=?,completion_tokens=?,total_tokens=?
                WHERE request_id=?""", (
                node, response.latency_ms, response.provider, response.actual_model,
                response.prompt_tokens, response.completion_tokens,
                response.total_tokens, request_id,
            ))
            spent, pending, uncertain = self._totals(db)
            if spent + pending + uncertain > self.limit:
                db.execute("UPDATE state SET stopped_reason='actual_over_budget' WHERE id=1")
            db.commit()
        if spent + pending + uncertain > self.limit:
            raise PilotBudgetStopped("实际用量超过本地预算，试跑已暂停")

    def stop(self, request_id: str, reason: str, latency_ms: float | None = None) -> None:
        # Only short internal reason codes are persisted, never provider error bodies.
        safe_reason = reason if reason in {
            "usage_missing", "usage_invalid", "model_mismatch", "http_402", "http_429", "request_uncertain"
        } else "request_uncertain"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Keep the worst-case reservation as a liability until cloud usage
            # is reconciled; an operator clearing the fuse cannot spend it twice.
            db.execute("""UPDATE calls SET status='uncertain',reason=?,latency_ms=?
                WHERE request_id=? AND status='reserved'""", (safe_reason, latency_ms, request_id))
            db.execute("UPDATE state SET stopped_reason=? WHERE id=1", (safe_reason,))
            db.commit()

    def summary(self) -> dict:
        with self._connect() as db:
            spent, reserved, uncertain = self._totals(db)
            reason = db.execute("SELECT stopped_reason FROM state WHERE id=1").fetchone()[0]
        return {
            "limit_tokens": self.limit,
            "spent_tokens": spent,
            "reserved_tokens": reserved,
            "uncertain_reserved_tokens": uncertain,
            "stopped": reason is not None,
            "stop_reason": reason,
        }
