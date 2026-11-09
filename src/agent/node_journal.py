"""Durable node results: reuse completed work, fence unknown outcomes."""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.exc import IntegrityError

from src.contracts import ContractError, ErrorCode


class NodeJournal:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        metadata = MetaData()
        self.table = Table("agent_node_results", metadata,
            Column("run_id", String(100), primary_key=True),
            Column("node", String(80), primary_key=True),
            Column("step", Integer, primary_key=True),
            Column("signature", String(64), nullable=False),
            Column("input_hash", String(64), nullable=False),
            Column("status", String(20), nullable=False),
            Column("output_json", Text),
            Column("error_type", String(80)))
        metadata.create_all(self.engine)

    def execute(self, run_id: str, node: str, step: int, signature: str,
                state: dict, operation: Callable[[], dict]) -> dict:
        key = (self.table.c.run_id == run_id, self.table.c.node == node,
               self.table.c.step == step)
        input_hash = hashlib.sha256(json.dumps(state, sort_keys=True,
            ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
        with self.engine.begin() as db:
            row = db.execute(select(self.table).where(*key)).mappings().first()
            if row is not None:
                if (row["signature"] != signature or row["input_hash"] != input_hash):
                    raise ContractError(ErrorCode.RUN_CONFLICT, "节点配置或输入已变化，不能恢复")
                if row["status"] != "completed":
                    raise ContractError(ErrorCode.RUN_CONFLICT, "节点结果不确定，需核查后重新提问")
                return json.loads(row["output_json"])
            try:
                db.execute(self.table.insert().values(run_id=run_id, node=node,
                    step=step, signature=signature, input_hash=input_hash, status="running"))
            except IntegrityError as exc:
                raise ContractError(ErrorCode.RUN_CONFLICT, "该节点已有执行者") from exc
        try:
            result = operation()
            output = json.dumps(result, ensure_ascii=False, default=lambda value: value.item())
            with self.engine.begin() as db:
                db.execute(self.table.update().where(*key).values(
                    status="completed", output_json=output))
            return result
        except Exception as exc:
            with self.engine.begin() as db:
                db.execute(self.table.update().where(*key).values(
                    status="uncertain", error_type=type(exc).__name__[:80]))
            raise

    def close(self) -> None:
        self.engine.dispose()
