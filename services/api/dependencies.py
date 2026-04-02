"""
services/api/dependencies.py

FastAPI dependency injection for:
- Database sessions (async SQLAlchemy)
- Current user (JWT auth)
- Storage client
- Queue client
"""
from __future__ import annotations

from typing import AsyncGenerator, Optional

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config.settings import settings

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────
_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

_async_session = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session per request."""
    async with _async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/token"
)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Validate JWT token and return user payload."""
    from jose import JWTError, jwt

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"],
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return {"username": username, "role": payload.get("role", "viewer")}
    except JWTError:
        raise credentials_exception


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_operator(user: dict = Depends(get_current_user)) -> dict:
    """Require operator or admin role."""
    if user.get("role") not in ("admin", "operator"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator access required",
        )
    return user


# ─────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────
def get_storage_client():
    """Return singleton MinIO storage client."""
    from core.abstractions.storage import get_storage
    return get_storage()


# ─────────────────────────────────────────────────────────────
# Queue
# ─────────────────────────────────────────────────────────────
def get_queue_client():
    """Return singleton queue backend."""
    from core.abstractions.queue import get_queue
    return get_queue()
