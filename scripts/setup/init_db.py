"""
scripts/setup/init_db.py
Initialize database: tables, pgvector, MinIO buckets, dan admin user.
Jalankan SEKALI setelah docker compose up pertama kali.
"""
from __future__ import annotations

import asyncio
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import structlog
log = structlog.get_logger(__name__)


async def init_database() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    import sqlalchemy as sa
    from core.config.settings import settings
    from core.models.database import Base

    log.info("db.init_start")
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(sa.text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    log.info("db.init_complete", message="All tables created")


async def init_vector_store() -> None:
    from core.abstractions.vector_store import get_vector_store
    store = get_vector_store()
    await store.create_collection("faces", dim=512)
    log.info("vector_store.initialized", collection="faces")


async def init_minio_buckets() -> None:
    from core.abstractions.storage import get_storage
    from core.config.settings import settings
    storage = get_storage()
    for bucket in [
        settings.MINIO_BUCKET_VIDEOS,
        settings.MINIO_BUCKET_FRAMES,
        settings.MINIO_BUCKET_FACES,
        settings.MINIO_BUCKET_COMPRESSED,
    ]:
        await storage.ensure_bucket(bucket)
        log.info("minio.bucket_ready", bucket=bucket)


async def create_admin_user() -> None:
    """
    Buat user admin default jika belum ada.
    Username: admin
    Password: changeme123  ← GANTI setelah pertama login!

    Menggunakan bcrypt langsung (bukan passlib) untuk kompatibilitas
    dengan bcrypt 4.x yang terinstall di container.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select
    from core.config.settings import settings
    from core.models.database import ApiUser

    try:
        import bcrypt as _bcrypt
    except ImportError:
        log.warning("admin.skip", reason="bcrypt not installed")
        return

    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "changeme123"
    DEFAULT_ROLE     = "admin"

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Cek apakah admin sudah ada
        result = await db.execute(
            select(ApiUser).where(ApiUser.username == DEFAULT_USERNAME)
        )
        existing = result.scalar_one_or_none()

        if existing:
            log.info("admin.already_exists", username=DEFAULT_USERNAME)
        else:
            # Generate bcrypt hash
            pw_hash = _bcrypt.hashpw(
                DEFAULT_PASSWORD.encode(),
                _bcrypt.gensalt()
            ).decode()

            admin_user = ApiUser(
                username=DEFAULT_USERNAME,
                hashed_password=pw_hash,
                role=DEFAULT_ROLE,
                is_active=True,
            )
            db.add(admin_user)
            await db.commit()
            log.info("admin.created",
                     username=DEFAULT_USERNAME,
                     role=DEFAULT_ROLE,
                     message="Login: admin / changeme123 — SEGERA GANTI PASSWORD!")

    await engine.dispose()


async def main() -> None:
    from core.config.logging import configure_logging
    configure_logging()

    log.info("setup.starting")
    try:
        await init_database()
        await init_vector_store()
        await init_minio_buckets()
        await create_admin_user()      # ← tambahan baru
        log.info("setup.complete", message="Platform initialized successfully")
    except Exception as e:
        log.error("setup.failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())