# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import secrets
from typing import Any, Optional

import bcrypt
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError
from jose import JWTError, jwt

from app.core.config import get_settings


_password_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1,
                                  hash_len=32, salt_len=16, type=Type.ID)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("$argon2id$"):
        try:
            return _password_hasher.verify(password_hash, password)
        except (InvalidHashError, VerificationError):
            return False
    if not password_hash.startswith(("$2a$", "$2b$", "$2y$")):
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def needs_password_rehash(password_hash: str) -> bool:
    return not password_hash.startswith("$argon2id$") or _password_hasher.check_needs_rehash(password_hash)


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def create_access_token(subject: str, extra: Optional[dict[str, Any]] = None) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expire, "jti": secrets.token_urlsafe(16)}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


class TokenError(Exception):
    pass


def safe_decode_token(token: str) -> dict[str, Any]:
    try:
        return decode_access_token(token)
    except JWTError as e:
        raise TokenError(str(e)) from e
