import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import documents
from app.core.deps import get_current_user
from app.db.base import Base
from app.db.schema_upgrades import run_schema_upgrades
from app.db.session import get_db


class DocumentsAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        run_schema_upgrades(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.current_user = SimpleNamespace(id=1, role="admin")

        app = FastAPI()
        app.include_router(documents.router, prefix="/api")

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

        self.settings_patch = patch(
            "app.api.documents.get_settings",
            return_value=SimpleNamespace(
                doc_artifact_root=str(self.tmp_path / "phase2")
            ),
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def test_normal_user_cannot_create_document(self):
        self.current_user.role = "user"
        response = self.client.post(
            "/api/documents",
            files={"file": ("rules.txt", "正文".encode("utf-8"), "text/plain")},
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_create_status_delete_and_duplicate(self):
        first = self.client.post(
            "/api/documents",
            files={"file": ("rules.txt", "正文".encode("utf-8"), "text/plain")},
        )
        self.assertEqual(first.status_code, 200)
        payload = first.json()
        document_id = payload["document_id"]
        version_id = payload["created_version_id"]

        status = self.client.get(f"/api/documents/{document_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "active")

        duplicate = self.client.post(
            f"/api/documents/{document_id}/versions",
            files={"file": ("rules.txt", "正文".encode("utf-8"), "text/plain")},
        )
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(duplicate.json()["created_version_id"], version_id)

        deleted = self.client.delete(f"/api/documents/{document_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["ok"], True)

    def test_invalid_file_returns_stable_code_without_absolute_path(self):
        response = self.client.post(
            "/api/documents",
            files={"file": ("report.pdf", b"MZfake", "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "signature_mismatch")
        self.assertNotIn("C:\\", detail["message"])


if __name__ == "__main__":
    unittest.main()
