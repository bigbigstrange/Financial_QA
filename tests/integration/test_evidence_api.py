import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import evidence
from app.core.deps import get_current_user
from app.db.base import Base
from app.db.session import get_db
from app.models import DocumentVersionRecord, EvidenceRecord, KnowledgeSnapshotRecord, User
from src.pipeline.snapshots import SnapshotManifest, SnapshotConfig, VersionArtifact, sha256_file, write_manifest_atomic
from src.pipeline.docir import DocBlock, DocIR, DocPage


class EvidenceAPITests(unittest.TestCase):
    def _add_image(self, *, linked=False, outside=False, invalid=False):
        from PIL import Image
        from src.system.visual_pages import make_visual_page_id
        doc = DocIR.from_dict(json.loads(self.docir_path.read_text(encoding='utf-8')))
        for n, color in [(1, 'red'), (2, 'blue')]:
            Image.new('RGB', (8, 8), color).save(self.tmp_path/f'{n}.png')
        if invalid:
            (self.tmp_path/'2.png').write_bytes(b'not an image')
        doc.pages = [DocPage(page_no=1, blocks=doc.pages[0].blocks, meta={'image_ref': '1.png'}),
                     DocPage(page_no=2, meta={'image_ref': '../outside.png' if outside else '2.png'})]
        self.docir_path.write_text(json.dumps(doc.to_dict()), encoding='utf-8')
        self._add_chunk()
        manifest = json.loads((self.tmp_path/'manifest.json').read_text(encoding='utf-8'))
        page_id = make_visual_page_id(manifest['snapshot_id'], 'doc-word', 'dv-word', 2)
        row = json.loads((self.tmp_path/'chunks.jsonl').read_text(encoding='utf-8'))
        row.update(page_start=1, page_end=2)
        (self.tmp_path/'chunks.jsonl').write_text(json.dumps(row), encoding='utf-8')
        with self.Session() as db:
            record = db.get(EvidenceRecord, 'ev-word')
            saved = json.loads(record.evidence_json)
            saved['metadata'].update(evidence_kind='text' if linked else 'page_image',
                vision_page=2, page_id=page_id, image_sha256=sha256_file(self.tmp_path/'2.png'))
            if not linked:
                record.block_id = page_id
                saved.update(text='', page_no=2)
                saved['metadata'].update(chunk_id=None, block_ids=[])
            record.evidence_json = json.dumps(saved)
            db.commit()
        return (self.tmp_path/'2.png').read_bytes()

    def test_pure_image_detail_and_exact_image_bytes(self):
        expected = self._add_image()
        response = self.client.get('/api/evidence/ev-word/image')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected)
        self.assertEqual(response.headers['content-type'], 'image/png')
        self.assertEqual(response.headers['cache-control'], 'private, no-store')
        self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
        detail = self.client.get('/api/evidence/ev-word').json()
        self.assertEqual(detail['text'], '')
        self.assertEqual(detail['evidence_kind'], 'page_image')
        self.assertEqual(detail['vision_page'], 2)
        self.assertTrue(detail['image_available'])

    def test_repeated_runs_and_users_preserve_each_visual_hit(self):
        from app.services.run_service import RunService
        from app.schemas import RunCreateRequest
        from src.contracts import KnowledgeSnapshotRef
        from src.system.visual_pages import make_visual_page_id
        from tests.unit.test_run_service import FakeRegistry
        self._add_image(linked=True)
        manifest = json.loads((self.tmp_path/'manifest.json').read_text(encoding='utf-8'))
        snapshot = manifest['snapshot_id']
        hit = dict(document_id='doc-word', document_version='dv-word', block_id='chk-word',
            chunk_id='chk-word', block_ids=['blk-word'], content='正文', page=1, snapshot_id=snapshot)
        service = RunService(registry=FakeRegistry(),
            knowledge_registry=SimpleNamespace(active_ref=lambda db: KnowledgeSnapshotRef(snapshot, str(self.tmp_path/'manifest.json'))),
            qa_runtime=SimpleNamespace(answer=lambda *args, **kw: {'answer':'候选', 'evidence':[dict(hit)]}))
        ids = []
        for owner, page in [(1, 1), (1, 2), (2, 2)]:
            hit.update(vision_page=page, page_id=make_visual_page_id(snapshot,'doc-word','dv-word',page),
                       image_sha256=sha256_file(self.tmp_path/f'{page}.png'))
            with self.Session() as db:
                result = service.create_and_execute(db, user_id=owner, request=RunCreateRequest(question='查看图片'))
            evidence_id = result.evidence[0].evidence_id
            ids.append(evidence_id)
            self.current_user = SimpleNamespace(id=owner, role='user')
            response = self.client.get(f'/api/evidence/{evidence_id}/image')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, (self.tmp_path/f'{page}.png').read_bytes())
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(self.client.get(f'/api/evidence/{ids[0]}/image').status_code, 404)
        self.current_user = SimpleNamespace(id=1, role='user')
        self.assertEqual(self.client.get(f'/api/evidence/{ids[0]}/image').content, (self.tmp_path/'1.png').read_bytes())

    def test_linked_chunk_image_is_hit_page_not_start_page(self):
        expected = self._add_image(linked=True)
        response = self.client.get('/api/evidence/ev-word/image')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected)
        detail = self.client.get('/api/evidence/ev-word').json()
        self.assertEqual((detail['page_no'], detail['page_end']), (1, 2))

    def test_image_cross_user_cannot_probe(self):
        self._add_image()
        self.current_user = SimpleNamespace(id=2, role='user')
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 404)

    def test_image_requires_authentication(self):
        del self.client.app.dependency_overrides[get_current_user]
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 401)

    def test_missing_image_returns_404(self):
        self._add_image()
        (self.tmp_path/'2.png').unlink()
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 404)

    def test_replaced_image_returns_409(self):
        self._add_image()
        (self.tmp_path/'2.png').write_bytes((self.tmp_path/'1.png').read_bytes())
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 409)

    def test_tampered_image_docir_returns_409(self):
        self._add_image()
        self.docir_path.write_text('{}', encoding='utf-8')
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 409)

    def test_tampered_page_id_returns_409(self):
        self._add_image()
        with self.Session() as db:
            record = db.get(EvidenceRecord, 'ev-word')
            record.block_id = 'vp_forged'
            db.commit()
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 409)

    def test_outside_image_path_returns_409_without_path_leak(self):
        self._add_image(outside=True)
        response = self.client.get('/api/evidence/ev-word/image')
        self.assertEqual(response.status_code, 409)
        self.assertNotIn(str(self.tmp_path), response.text)

    def test_non_image_bytes_return_409(self):
        self._add_image(invalid=True)
        self.assertEqual(self.client.get('/api/evidence/ev-word/image').status_code, 409)

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

        with self.Session() as db:
            self.owner = User(
                id=1,
                username="owner",
                email="owner@example.com",
                password_hash="x",
                role="user",
            )
            self.other = User(
                id=2,
                username="other",
                email="other@example.com",
                password_hash="x",
                role="user",
            )
            self.admin = User(
                id=3,
                username="admin",
                email="admin@example.com",
                password_hash="x",
                role="admin",
            )
            db.add_all([self.owner, self.other, self.admin])

            doc = DocIR(
                doc_id="doc-word",
                source_path="mixed.docx",
                source_format=".docx",
                document_version="dv-word",
                source_name="mixed.docx",
                schema_version=2,
            )
            doc.add_page(
                DocPage(
                    page_no=None,
                    blocks=[
                        DocBlock(
                            "blk-word",
                            "paragraph",
                            "正文",
                            None,
                            section_path="第一章",
                            structural_locator="body/000003/paragraph",
                        )
                    ],
                )
            )
            self.docir_path = self.tmp_path / "dv-word.docir.json"
            self.docir_path.write_text(
                json.dumps(doc.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            version = DocumentVersionRecord(
                id="dv-word",
                document_id="doc-word",
                source_sha256="a",
                source_size_bytes=1,
                source_format=".docx",
                parser_name="word_structured",
                parser_version="2",
                parser_config_hash="c",
                chunker_version="chunk-v2",
                status="active",
                docir_path=str(self.docir_path),
                quality_report_path="quality.json",
            )
            record = EvidenceRecord(
                id="ev-word",
                run_id=None,
                user_id=1,
                document_id="doc-word",
                document_version_id="dv-word",
                block_id="blk-word",
                evidence_json="{}",
            )
            db.add_all([version, record])
            db.commit()
            self.version = version
            self.evidence = record

        self.current_user = SimpleNamespace(id=1, role="user")
        app = FastAPI()
        app.include_router(evidence.router, prefix="/api")

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        def override_user():
            return self.current_user

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_current_user] = override_user
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.temp_dir.cleanup)

    def test_owner_reads_word_structure_without_fake_page(self):
        response = self.client.get("/api/evidence/ev-word")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["page_no"])
        self.assertEqual(payload["section_path"], "第一章")
        self.assertEqual(payload["structural_locator"], "body/000003/paragraph")

    def test_other_user_cannot_probe_evidence_id(self):
        self.current_user = SimpleNamespace(id=2, role="user")
        response = self.client.get("/api/evidence/ev-word")

        self.assertEqual(response.status_code, 404)

    def test_admin_can_read_after_version_is_deleted(self):
        with self.Session() as db:
            version = db.get(DocumentVersionRecord, "dv-word")
            version.status = "deleted"
            db.commit()
        self.current_user = SimpleNamespace(id=3, role="admin")

        response = self.client.get("/api/evidence/ev-word")

        self.assertEqual(response.status_code, 200)

    def test_block_identity_tampering_is_rejected(self):
        with self.Session() as db:
            record = db.get(EvidenceRecord, "ev-word")
            record.block_id = "blk-tampered"
            db.commit()

        response = self.client.get("/api/evidence/ev-word")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"]["code"],
            "evidence_identity_mismatch",
        )

    def _add_chunk(self):
        manifest = SnapshotManifest.build([VersionArtifact("doc-word", "dv-word", str(self.docir_path), 1,
            sha256_file(self.docir_path))], SnapshotConfig(3, "chunker-v3", "test"))
        path = self.tmp_path / "manifest.json"
        write_manifest_atomic(manifest, path)
        row = {"chunk_id": "chk-word", "document_version": "dv-word", "snapshot_id": manifest.snapshot_id,
               "block_ids": ["blk-word"], "content": "正文", "page_start": None}
        (self.tmp_path / "chunks.jsonl").write_text(json.dumps(row), encoding="utf-8")
        with self.Session() as db:
            db.add(KnowledgeSnapshotRecord(id=manifest.snapshot_id, manifest_path=str(path),
                manifest_sha256=manifest.manifest_sha256, document_versions_json='["dv-word"]'))
            rec = db.get(EvidenceRecord, "ev-word")
            rec.block_id = "chk-word"
            rec.evidence_json = json.dumps({"text": "正文", "metadata": {"snapshot_id": manifest.snapshot_id}})
            db.commit()

    def test_chunk_resolves_original_blocks_without_fake_page(self):
        self._add_chunk()
        response = self.client.get("/api/evidence/ev-word")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "正文")
        self.assertIsNone(response.json()["page_no"])

    def test_chunk_content_tampering_is_rejected(self):
        self._add_chunk()
        path = self.tmp_path / "chunks.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["content"] = "伪造"
        path.write_text(json.dumps(row), encoding="utf-8")
        self.assertEqual(self.client.get("/api/evidence/ev-word").status_code, 409)


if __name__ == "__main__":
    unittest.main()
