import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password, verify_password
from app.db import init_db as bootstrap
from app.db.base import Base
from app.models import User


class AdminBootstrapTests(unittest.TestCase):
    def test_production_rotates_existing_default_admin_password(self):
        engine = create_engine("sqlite://", poolclass=StaticPool)
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            db.add(User(username="admin", email="admin@example.com", password_hash=hash_password("admin123"), role="admin"))
            db.commit()

        settings = SimpleNamespace(
            app_env="production",
            bootstrap_admin_user="admin",
            bootstrap_admin_email="admin@example.com",
            bootstrap_admin_password="new-random-password-123456",
        )
        with patch.object(bootstrap, "engine", engine), patch.object(bootstrap, "SessionLocal", sessions), patch.object(bootstrap, "get_settings", return_value=settings):
            bootstrap.init_db()

        with sessions() as db:
            admin = db.query(User).filter(User.username == "admin").one()
            self.assertFalse(verify_password("admin123", admin.password_hash))
            self.assertTrue(verify_password("new-random-password-123456", admin.password_hash))


if __name__ == "__main__":
    unittest.main()
