"""
scripts/setup/init_db.py
Initialize database: create tables, pgvector extension, seed initial data.
Run once after first docker compose up.
"""
from __future__ import annotations

import asyncio
import sys

import structlog

log = structlog.get_logger(__name__)


async def init_database() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    import sqlalchemy as sa
    from core.config.settings import settings
    from core.models.database import Base

    log.info("db.init_start")
    engine = create_async_engine(settings.DATABASE_URL, echo=True)

    async with engine.begin() as conn:
        # Enable pgvector
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(sa.text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))

        # Create all tables from ORM models
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


async def main() -> None:
    from core.config.logging import configure_logging
    configure_logging()

    log.info("setup.starting")
    try:
        await init_database()
        await init_vector_store()
        await init_minio_buckets()
        log.info("setup.complete", message="Platform initialized successfully")
    except Exception as e:
        log.error("setup.failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
