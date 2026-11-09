# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Annotated
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.config import get_settings
from app.core.security import (as_utc, create_access_token, hash_password,
                               needs_password_rehash, verify_password)
from app.db.session import get_db
from app.models import RefreshSession, User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
REFRESH_COOKIE = "finance_refresh"
CSRF_COOKIE = "finance_csrf"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _set_session_cookies(response: Response, token: str, csrf: str) -> None:
    settings = get_settings()
    max_age = settings.refresh_token_expire_days * 86400
    secure = settings.app_env == "production"
    response.set_cookie(REFRESH_COOKIE, token, httponly=True, secure=secure,
                        samesite="lax", path="/api/auth", max_age=max_age)
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, secure=secure,
                        samesite="lax", path="/", max_age=max_age)


def _clear_session_cookies(response: Response) -> None:
    response.delete_cookie(REFRESH_COOKIE, path="/api/auth")
    response.delete_cookie(CSRF_COOKIE, path="/")


def _validated_session(request: Request, db: Session) -> RefreshSession:
    origin = request.headers.get("origin")
    same_origin = f"{request.url.scheme}://{request.url.netloc}"
    if not origin or origin not in {same_origin, *get_settings().cors_origin_list}:
        raise HTTPException(status_code=403, detail="请求来源未获授权")
    token = request.cookies.get(REFRESH_COOKIE) or ""
    csrf = request.headers.get("X-CSRF-Token") or ""
    if not token or not csrf or not hmac.compare_digest(csrf, request.cookies.get(CSRF_COOKIE) or ""):
        raise HTTPException(status_code=403, detail="缺少有效的 CSRF 校验")
    row = db.query(RefreshSession).filter_by(token_hash=_digest(token)).first()
    now = datetime.now(timezone.utc)
    if (row is None or row.revoked_at is not None or
            as_utc(row.expires_at) <= now or
            not hmac.compare_digest(_digest(csrf), row.csrf_hash)):
        raise HTTPException(status_code=401, detail="刷新会话已失效")
    return row


@router.post("/register", response_model=UserOut)
def register(body: RegisterRequest, db: Annotated[Session, Depends(get_db)]) -> User:
    if db.query(User).filter((User.username == body.username) | (User.email == body.email)).first():
        raise HTTPException(status_code=400, detail="用户名或邮箱已存在")
    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, response: Response,
          db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if needs_password_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        db.commit()
    refresh = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    session_id = str(uuid.uuid4())
    db.add(RefreshSession(id=session_id, user_id=user.id, token_hash=_digest(refresh),
        csrf_hash=_digest(csrf), expires_at=datetime.now(timezone.utc) +
        timedelta(days=get_settings().refresh_token_expire_days)))
    db.commit()
    _set_session_cookies(response, refresh, csrf)
    token = create_access_token(user.username, extra={"role": user.role, "sid": session_id})
    return TokenResponse(access_token=token)


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: Request, response: Response,
            db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    row = _validated_session(request, db)
    user = db.get(User, row.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    old_hash = row.token_hash
    new_refresh = secrets.token_urlsafe(32)
    new_csrf = secrets.token_urlsafe(32)
    changed = db.execute(update(RefreshSession)
        .where(RefreshSession.id == row.id, RefreshSession.token_hash == old_hash,
               RefreshSession.revoked_at.is_(None))
        .values(token_hash=_digest(new_refresh), csrf_hash=_digest(new_csrf),
                expires_at=datetime.now(timezone.utc) +
                timedelta(days=get_settings().refresh_token_expire_days))).rowcount
    if not changed:
        db.rollback()
        raise HTTPException(status_code=401, detail="刷新会话已失效")
    db.commit()
    _set_session_cookies(response, new_refresh, new_csrf)
    return TokenResponse(access_token=create_access_token(user.username,
        extra={"role": user.role, "sid": row.id}))


@router.post("/logout")
def logout(request: Request, response: Response,
           db: Annotated[Session, Depends(get_db)]) -> dict[str, bool]:
    row = _validated_session(request, db)
    db.execute(update(RefreshSession).where(RefreshSession.id == row.id,
        RefreshSession.revoked_at.is_(None)).values(revoked_at=datetime.now(timezone.utc)))
    db.commit()
    _clear_session_cookies(response)
    return {"ok": True}


@router.get("/me", response_model=UserOut)
def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user
