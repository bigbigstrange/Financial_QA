# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


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
def login(body: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token(user.username, extra={"role": user.role})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserOut)
def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user
