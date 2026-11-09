import bcrypt
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth import router
from app.core.security import hash_password, needs_password_rehash, verify_password
from app.db.base import Base
from app.db.session import get_db
from app.models import User


def test_argon2id_new_hash_and_legacy_bcrypt_verification():
    modern = hash_password("safe-demo-password")
    legacy = bcrypt.hashpw(b"old-password", bcrypt.gensalt()).decode()
    assert modern.startswith("$argon2id$")
    assert verify_password("safe-demo-password", modern)
    assert not verify_password("wrong", modern)
    assert not needs_password_rehash(modern)
    assert verify_password("old-password", legacy)
    assert not verify_password("wrong", legacy)
    assert needs_password_rehash(legacy)
    assert not verify_password("x", "malformed-hash")


def test_successful_login_upgrades_bcrypt_without_reset():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    legacy = bcrypt.hashpw(b"old-password", bcrypt.gensalt()).decode()
    with Session() as db:
        db.add(User(username="existing", email="existing@example.com",
                    password_hash=legacy, role="user"))
        db.commit()
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    with TestClient(app) as client:
        assert client.post("/api/auth/login", json={"username": "existing",
            "password": "wrong"}).status_code == 401
        with Session() as db:
            assert db.query(User).filter_by(username="existing").one().password_hash == legacy
        response = client.post("/api/auth/login", json={"username": "existing",
            "password": "old-password"})
        assert response.status_code == 200
    with Session() as db:
        upgraded = db.query(User).filter_by(username="existing").one().password_hash
        assert upgraded.startswith("$argon2id$")
        assert verify_password("old-password", upgraded)
    engine.dispose()


def test_refresh_rotates_cookie_checks_csrf_and_logout_revokes_access():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        db.add(User(username="session-user", email="session@example.com",
                    password_hash=hash_password("safe-demo-password"), role="user"))
        db.commit()
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: Session()
    with TestClient(app) as client:
        login = client.post("/api/auth/login", json={"username": "session-user",
            "password": "safe-demo-password"})
        assert login.status_code == 200
        access = login.json()["access_token"]
        assert "httponly" in login.headers["set-cookie"].lower()
        csrf = client.cookies.get("finance_csrf")
        assert csrf
        assert client.post("/api/auth/refresh", headers={"Origin": "http://testserver",
            "X-CSRF-Token": "wrong"}).status_code == 403
        assert client.post("/api/auth/refresh", headers={"Origin": "https://evil.example",
            "X-CSRF-Token": csrf}).status_code == 403
        renewed = client.post("/api/auth/refresh", headers={"Origin": "http://testserver",
            "X-CSRF-Token": csrf})
        assert renewed.status_code == 200
        assert renewed.json()["access_token"] != access
        assert client.post("/api/auth/refresh", headers={"Origin": "http://testserver",
            "X-CSRF-Token": csrf}).status_code == 403
        assert client.get("/api/auth/me", headers={"Authorization":
            f"Bearer {renewed.json()['access_token']}"}).status_code == 200
        new_csrf = client.cookies.get("finance_csrf")
        logged_out = client.post("/api/auth/logout", headers={"Origin": "http://testserver",
            "X-CSRF-Token": new_csrf})
        assert logged_out.status_code == 200
        assert client.get("/api/auth/me", headers={"Authorization":
            f"Bearer {renewed.json()['access_token']}"}).status_code == 401
    engine.dispose()
