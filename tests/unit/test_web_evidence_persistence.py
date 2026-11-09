from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import EvidenceRecord
from app.schemas import RunCreateRequest
from app.services.evidence_reader import EvidenceNotFoundError, EvidenceReader
from app.services.run_service import RunService
from src.contracts import ModelSnapshot, RunStatus


class Registry:
    def snapshot(self):
        return ModelSnapshot(snapshot_id="test", profile_name="test", provider="fake")


class WebQA:
    def answer(self, question, question_id, knowledge_snapshot=None):
        return {
            "answer": "依据公开网页[1]。",
            "verified": True,
            "evidence": [{
                "source_doc": "中国人民银行公告",
                "source_url": "https://www.pbc.gov.cn/example",
                "content": "公告摘录",
                "retrieval_channels": ["web"],
            }],
        }


def test_web_evidence_persists_and_reads_on_historical_local_schema():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    with engine.begin() as connection:
        connection.execute(text("""CREATE TABLE evidence_records (
            id VARCHAR(80) NOT NULL, run_id VARCHAR(36) NOT NULL,
            user_id INTEGER NOT NULL, snapshot_id VARCHAR(64) NOT NULL,
            document_id VARCHAR(64) NOT NULL, document_version VARCHAR(64) NOT NULL,
            block_id VARCHAR(80) NOT NULL, quote TEXT NOT NULL,
            scores_json TEXT NOT NULL, document_version_id VARCHAR(64),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (id, run_id))"""))
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        result = RunService(qa_runtime=WebQA(), registry=Registry()).create_and_execute(
            db, user_id=7, request=RunCreateRequest(question="央行公告是什么？"))
        assert result.status == RunStatus.COMPLETED
        assert result.evidence[0].document_id is None
        record = db.get(EvidenceRecord, result.evidence[0].evidence_id)
        assert record.block_id.startswith("web:")
        assert record.document_version_id is None
        detail = EvidenceReader.read(db, record.id, SimpleNamespace(id=7, role="user"))
        assert detail["evidence_kind"] == "web"
        assert detail["source_url"] == "https://www.pbc.gov.cn/example"
        assert detail["text"] == "公告摘录"
        assert detail["document_version"] is None
        with pytest.raises(EvidenceNotFoundError):
            EvidenceReader.read(db, record.id, SimpleNamespace(id=8, role="user"))
    engine.dispose()
