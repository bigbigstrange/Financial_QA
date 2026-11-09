"""Idempotent additive schema upgrades for an existing phase-1 database."""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def run_schema_upgrades(engine: Engine) -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if "runs" in tables:
            columns = {column["name"] for column in inspector.get_columns("runs")}
            if "knowledge_snapshot_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE runs "
                        "ADD COLUMN knowledge_snapshot_id VARCHAR(64)"
                    )
                )
            if "agent_trace_json" not in columns:
                # 阶段 C：Agent 决策轨迹落库，供 /runs/{id}/trace 回放
                connection.execute(
                    text("ALTER TABLE runs ADD COLUMN agent_trace_json TEXT")
                )
            if "checkpoint_json" not in columns:
                connection.execute(text("ALTER TABLE runs ADD COLUMN checkpoint_json TEXT"))
            if "cancel_requested_at" not in columns:
                date_type = "TIMESTAMP" if engine.dialect.name == "postgresql" else "DATETIME"
                connection.execute(text(f"ALTER TABLE runs ADD COLUMN cancel_requested_at {date_type}"))
            if "options_json" not in columns:
                connection.execute(text("ALTER TABLE runs ADD COLUMN options_json TEXT NOT NULL DEFAULT '{}'"))
        if "evidence_records" in tables:
            columns = {c["name"] for c in inspector.get_columns("evidence_records")}
            if "document_version_id" not in columns:
                connection.execute(text("ALTER TABLE evidence_records ADD COLUMN document_version_id VARCHAR(64)"))
                if "document_version" in columns:
                    connection.execute(text("UPDATE evidence_records SET document_version_id = document_version"))
            if "evidence_json" not in columns:
                connection.execute(text("ALTER TABLE evidence_records ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '{}'"))
