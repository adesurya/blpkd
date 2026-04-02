"""
services/api/main.py
FastAPI application entry point.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from core.config.settings import settings
from core.config.logging import configure_logging
from services.api.routers import streams, videos, faces, analytics, webhooks, auth

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    configure_logging()
    log.info("app.starting", version=settings.APP_VERSION, env=settings.APP_ENV)

    # Initialize vector store collections
    from core.abstractions.vector_store import get_vector_store
    store = get_vector_store()
    await store.create_collection("faces", dim=512)

    # Warm up AI models (optional, comment out if slow startup is OK)
    # from services.detector.yolo_detector import YOLODetector
    # YOLODetector.get_instance().warmup()

    log.info("app.ready")
    yield

    log.info("app.shutting_down")
    from services.video_processor.stream_reader import StreamManager
    # Stop all active streams
    manager = app.state.stream_manager if hasattr(app.state, "stream_manager") else None
    if manager:
        await manager.stop_all()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vision Platform API",
        description="AI Vision, Face Recognition & Video Analytics Backend",
        version=settings.APP_VERSION,
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Prometheus Metrics ──────────────────────────────────────
    if settings.PROMETHEUS_ENABLED:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Routers ─────────────────────────────────────────────────
    prefix = settings.API_V1_PREFIX
    app.include_router(auth.router,      prefix=prefix, tags=["Authentication"])
    app.include_router(streams.router,   prefix=prefix, tags=["Streams & Cameras"])
    app.include_router(videos.router,    prefix=prefix, tags=["Video Processing"])
    app.include_router(faces.router,     prefix=prefix, tags=["Face Recognition"])
    app.include_router(analytics.router, prefix=prefix, tags=["Analytics & Counting"])
    app.include_router(webhooks.router,  prefix=prefix, tags=["Webhooks"])

    # ── Health Check ────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "env": settings.APP_ENV,
        }

    @app.get("/health/detailed", tags=["System"])
    async def detailed_health():
        import redis.asyncio as aioredis
        checks = {}

        # Redis
        try:
            r = aioredis.from_url(settings.REDIS_URL)
            await r.ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {e}"

        # PostgreSQL
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            engine = create_async_engine(settings.DATABASE_URL)
            async with engine.connect() as conn:
                await conn.execute(sqlalchemy.text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception as e:
            checks["postgres"] = f"error: {e}"

        # MinIO
        try:
            from minio import Minio
            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            list(client.list_buckets())
            checks["minio"] = "ok"
        except Exception as e:
            checks["minio"] = f"error: {e}"

        all_ok = all(v == "ok" for v in checks.values())
        return {
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
        }

    return app


app = create_app()
