"""
core/config/settings.py
Centralized configuration using Pydantic Settings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────
    APP_NAME: str = "vision-platform"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_VERSION: str = "0.1.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = Field(min_length=32)

    # ── Database ───────────────────────────────────────────
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "visiondb"
    POSTGRES_USER: str = "visionuser"
    POSTGRES_PASSWORD: str = "visionpass"

    # ── Redis ──────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    # ── MinIO ──────────────────────────────────────────────
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_VIDEOS: str = "videos"
    MINIO_BUCKET_FRAMES: str = "frames"
    MINIO_BUCKET_FACES: str = "faces"
    MINIO_BUCKET_COMPRESSED: str = "compressed"

    # ── AI Models ──────────────────────────────────────────
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45
    # Default cpu — override ke "cuda" di .env jika pakai GPU
    YOLO_DEVICE: str = "cpu"

    INSIGHTFACE_MODEL: str = "buffalo_l"
    INSIGHTFACE_DEVICE: str = "cpu"
    FACE_RECOGNITION_THRESHOLD: float = 0.45
    FACE_DETECTION_MIN_SIZE: int = 20

    DBSCAN_EPS: float = 0.6
    DBSCAN_MIN_SAMPLES: int = 2

    # ── Video Processing ───────────────────────────────────
    FRAME_SAMPLE_RATE: int = 5
    MAX_FRAME_WIDTH: int = 1920
    MAX_FRAME_HEIGHT: int = 1080
    FRAME_FORMAT: str = "jpg"

    STREAM_RECONNECT_DELAY: int = 5
    STREAM_MAX_RETRIES: int = 10
    STREAM_BUFFER_SIZE: int = 10

    # ── Compression ────────────────────────────────────────
    # Default libx264 (CPU) — override ke h264_nvenc di .env jika pakai GPU
    COMPRESSION_CODEC: str = "libx264"
    COMPRESSION_PRESET: str = "medium"
    COMPRESSION_CRF: int = 23
    COMPRESSION_AUDIO_CODEC: str = "aac"
    COMPRESSION_AUDIO_BITRATE: str = "128k"
    GPU_DECODE: bool = False

    # ── API ────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    API_CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    API_RATE_LIMIT: int = 100
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ── Paths ──────────────────────────────────────────────
    MODELS_PATH: str = "/app/models"
    TEMP_PATH: str = "/tmp/vision"
    LOG_PATH: str = "/app/logs"

    # ── Monitoring ─────────────────────────────────────────
    PROMETHEUS_ENABLED: bool = True

    # ── Webhook ────────────────────────────────────────────
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_RETRY_DELAY: int = 5

    # ── Scale — queue backend ──────────────────────────────
    QUEUE_BACKEND: Literal["redis", "rabbitmq", "kafka"] = "redis"
    RABBITMQ_URL: str = "amqp://guest:guest@rabbitmq:5672/"
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"

    # ── Scale — vector db backend ──────────────────────────
    VECTOR_BACKEND: Literal["pgvector", "milvus"] = "pgvector"
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: int = 19530

    @field_validator("YOLO_DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """
        Validasi device tanpa import torch.
        torch hanya tersedia di GPU worker — di container lain cukup
        terima nilai dari .env apa adanya.
        Jika torch tidak ada dan nilai adalah 'cuda', fallback ke 'cpu'.
        """
        if not v.startswith("cuda"):
            return v

        # Coba cek torch, tapi tidak crash jika tidak ada
        try:
            import torch
            if not torch.cuda.is_available():
                import structlog
                structlog.get_logger().warning(
                    "CUDA not available, falling back to CPU",
                    requested_device=v
                )
                return "cpu"
        except ImportError:
            # torch tidak terinstall di container ini (API / CPU worker)
            # Kembalikan "cpu" agar tidak ada error
            return "cpu"

        return v

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache
def get_settings() -> AppSettings:
    """Cached settings singleton — call get_settings() anywhere."""
    return AppSettings()


# Convenience export
settings = get_settings()