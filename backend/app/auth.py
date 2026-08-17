"""Password hashing, JWT issue/verify, and the `get_current_user` dependency.

The authenticated user_id is ALWAYS taken from the token — never from a request body or query
param. Every router outside /auth depends on `get_current_user`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(raw: str) -> str:
    # bcrypt silently truncates at 72 bytes; reject rather than accept a weaker password than typed.
    if len(raw.encode("utf-8")) > 72:
        raise HTTPException(status_code=422, detail="Password must be at most 72 bytes")
    return _pwd.hash(raw)


def verify_password(raw: str, hashed: str) -> bool:
    if len(raw.encode("utf-8")) > 72:
        return False
    return _pwd.verify(raw, hashed)


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return int(payload["sub"])
    except (JWTError, KeyError, TypeError, ValueError) as exc:
        raise CREDENTIALS_ERROR from exc


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None or not creds.credentials:
        raise CREDENTIALS_ERROR
    user = db.get(User, decode_token(creds.credentials))
    if user is None:
        raise CREDENTIALS_ERROR
    return user
