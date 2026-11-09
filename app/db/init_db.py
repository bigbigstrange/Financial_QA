# -*- coding: utf-8 -*-
from __future__ import annotations

from app.core.config import get_settings
from app.core.security import hash_password, verify_password
from app.db.base import Base
from app.db.schema_upgrades import run_schema_upgrades
from app.db.session import SessionLocal, engine
from app.models import User


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    run_schema_upgrades(engine)
    settings = get_settings()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == settings.bootstrap_admin_user).first()
        if not admin:
            db.add(
                User(
                    username=settings.bootstrap_admin_user,
                    email=settings.bootstrap_admin_email,
                    password_hash=hash_password(settings.bootstrap_admin_password),
                    role="admin",
                )
            )
            db.commit()
        else:
            if settings.app_env == "production" and verify_password("admin123", admin.password_hash):
                admin.password_hash = hash_password(settings.bootstrap_admin_password)
            if admin.email.endswith(".local") or "@finance-qa." in admin.email:
                # 兼容旧种子邮箱，避免响应校验失败
                admin.email = settings.bootstrap_admin_email
            if db.is_modified(admin):
                db.commit()
    finally:
        db.close()
