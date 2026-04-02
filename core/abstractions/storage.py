"""
core/abstractions/storage.py

Object storage abstraction — swap MinIO single-node → MinIO cluster → AWS S3
by changing env vars. No business logic changes needed.

All APIs are S3-compatible, so MinIO and AWS S3 use the same code path.
"""
from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import BinaryIO, Optional

import structlog

log = structlog.get_logger(__name__)


@dataclass
class StoredObject:
    bucket: str
    key: str
    size_bytes: int
    content_type: str
    url: Optional[str] = None       # presigned URL if requested


class ObjectStorage(ABC):
    """Abstract object storage interface."""

    @abstractmethod
    async def upload(
        self,
        bucket: str,
        key: str,
        data: BinaryIO | bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> StoredObject:
        """Upload an object. Returns StoredObject with key."""
        ...

    @abstractmethod
    async def download(self, bucket: str, key: str) -> bytes:
        """Download and return object bytes."""
        ...

    @abstractmethod
    async def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a time-limited presigned URL for direct access."""
        ...

    @abstractmethod
    async def delete(self, bucket: str, key: str) -> None:
        """Delete an object."""
        ...

    @abstractmethod
    async def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists."""
        ...

    @abstractmethod
    async def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """List object keys with optional prefix."""
        ...

    @abstractmethod
    async def ensure_bucket(self, bucket: str) -> None:
        """Create bucket if it doesn't exist."""
        ...


class MinIOStorage(ObjectStorage):
    """
    MinIO / S3-compatible object storage.
    Works for: MinIO (single or cluster), AWS S3, GCS (via compatibility layer).

    MVP:        Single MinIO node  (docker compose)
    Growth:     MinIO 3-node cluster
    Enterprise: AWS S3 or MinIO on Kubernetes
    Only change: endpoint + credentials in .env
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ) -> None:
        from minio import Minio
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        log.info("storage.initialized", endpoint=endpoint, secure=secure)

    async def ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            log.info("storage.bucket_created", bucket=bucket)

    async def upload(
        self,
        bucket: str,
        key: str,
        data: BinaryIO | bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> StoredObject:
        await self.ensure_bucket(bucket)

        if isinstance(data, bytes):
            stream = io.BytesIO(data)
            size = len(data)
        else:
            # File-like object — read to get size, then seek back
            content = data.read()
            stream = io.BytesIO(content)
            size = len(content)

        self._client.put_object(
            bucket,
            key,
            stream,
            length=size,
            content_type=content_type,
            metadata=metadata,
        )

        log.debug("storage.uploaded", bucket=bucket, key=key, size=size)
        return StoredObject(bucket=bucket, key=key, size_bytes=size, content_type=content_type)

    async def download(self, bucket: str, key: str) -> bytes:
        response = self._client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        url = self._client.presigned_get_object(
            bucket,
            key,
            expires=timedelta(seconds=expires_in),
        )
        return url

    async def delete(self, bucket: str, key: str) -> None:
        self._client.remove_object(bucket, key)
        log.debug("storage.deleted", bucket=bucket, key=key)

    async def exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.stat_object(bucket, key)
            return True
        except Exception:
            return False

    async def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        objects = self._client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]


# ─────────────────────────────────────────────────────────────
# Storage key helpers — consistent naming convention
# ─────────────────────────────────────────────────────────────
class StorageKeys:
    """
    Centralized key naming convention for all stored objects.
    Ensures consistent paths across the application.
    """

    @staticmethod
    def video_original(recording_id: str) -> str:
        return f"original/{recording_id}.mp4"

    @staticmethod
    def video_compressed(recording_id: str) -> str:
        return f"compressed/{recording_id}.mp4"

    @staticmethod
    def frame(session_id: str, frame_number: int) -> str:
        return f"{session_id}/frames/frame_{frame_number:08d}.jpg"

    @staticmethod
    def face_crop(face_id: str) -> str:
        return f"crops/{face_id[:2]}/{face_id}.jpg"

    @staticmethod
    def person_crop(session_id: str, track_id: int, frame_number: int) -> str:
        return f"{session_id}/persons/track_{track_id}_frame_{frame_number:08d}.jpg"


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────
_storage_instance: Optional[ObjectStorage] = None


def get_storage() -> ObjectStorage:
    """Singleton storage instance."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    from core.config.settings import settings
    _storage_instance = MinIOStorage(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )
    return _storage_instance
