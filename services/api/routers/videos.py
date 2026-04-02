"""
services/api/routers/videos.py
Video upload, processing, and compression — full implementation.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.abstractions.storage import get_storage, StorageKeys
from core.models.repository import RecordingRepository
from services.api.dependencies import get_db, get_current_user
from services.api.routers.schemas import (
    VideoUploadResponse, VideoProcessingResult, TaskStatusResponse,
)

router = APIRouter()
log = structlog.get_logger(__name__)
UTC = timezone.utc

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB


@router.post("/videos/upload", response_model=VideoUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    extract_faces: bool = Form(True),
    analyze_attributes: bool = Form(True),
    compress: bool = Form(True),
    sample_rate: int = Form(5),
    zones: Optional[str] = Form(None),
    filter_criteria: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Upload a video file for AI analysis and optional compression.

    Returns immediately with `recording_id` and `task_id`.
    Poll `GET /videos/{recording_id}/status` for progress.

    **Processing pipeline:**
    1. Stream upload → MinIO (original bucket)
    2. YOLOv8 people detection on sampled frames
    3. InsightFace face recognition on person crops
    4. Clothing color & attribute analysis
    5. Save all results to PostgreSQL
    6. FFmpeg NVENC compression → MinIO (compressed bucket)
    """
    import json
    from workers.detection_tasks import process_video_file

    # Validate file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    recording_id = str(uuid.uuid4())
    storage = get_storage()

    # Stream file to MinIO
    log.info("video.upload_start", recording_id=recording_id, filename=filename)
    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"File too large. Max size: 5GB")

        obj = await storage.upload(
            bucket="videos",
            key=StorageKeys.video_original(recording_id),
            data=content,
            content_type=file.content_type or "video/mp4",
            metadata={"original_filename": filename, "recording_id": recording_id},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("video.upload_failed", error=str(e))
        raise HTTPException(500, f"Upload failed: {e}")

    # Save recording metadata to DB
    rec = await RecordingRepository.create(db, {
        "camera_id": camera_id,
        "source_type": "upload",
        "original_path": StorageKeys.video_original(recording_id),
        "original_size_bytes": obj.size_bytes,
        "status": "queued",
    })

    # Parse optional JSON params
    zones_dict = json.loads(zones) if zones else None
    filter_dict = json.loads(filter_criteria) if filter_criteria else None

    task_config = {
        "camera_id": camera_id,
        "extract_faces": extract_faces,
        "analyze_attributes": analyze_attributes,
        "compress": compress,
        "sample_rate": sample_rate,
        "zones": zones_dict,
        "filter_criteria": filter_dict,
        "recording_id": recording_id,
    }

    # Queue Celery task
    task = process_video_file.apply_async(
        args=[recording_id, StorageKeys.video_original(recording_id), task_config],
        queue="gpu_tasks",
    )

    # Update DB with task ID
    await RecordingRepository.update_status(
        db, recording_id, "queued", {"processing_task_id": task.id}
    )

    log.info("video.upload_complete", recording_id=recording_id, task_id=task.id,
             size_mb=round(obj.size_bytes / 1024 / 1024, 1))

    return VideoUploadResponse(
        recording_id=recording_id,
        task_id=task.id,
        status="queued",
        message="Video queued. Poll /videos/{recording_id}/status for updates.",
    )


@router.get("/videos/{recording_id}/status", response_model=TaskStatusResponse)
async def get_processing_status(
    recording_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Poll video processing status.

    **status**: `pending` | `running` | `completed` | `failed`
    """
    from workers.celery_app import celery_app
    from celery.result import AsyncResult

    rec = await RecordingRepository.get(db, recording_id)
    if not rec:
        raise HTTPException(404, f"Recording '{recording_id}' not found")

    task_result: Optional[dict] = None
    task_error: Optional[str] = None
    progress: Optional[float] = None

    if rec.processing_task_id:
        try:
            result = AsyncResult(rec.processing_task_id, app=celery_app)
            if result.state == "SUCCESS":
                task_result = result.result
                progress = 1.0
            elif result.state == "FAILURE":
                task_error = str(result.result)
            elif result.state in ("STARTED", "PROGRESS"):
                info = result.info or {}
                progress = info.get("progress", 0.0)
        except Exception:
            pass

    # Map celery status to our status vocabulary
    status_map = {
        "uploaded": "pending",
        "queued": "pending",
        "processing": "running",
        "completed": "completed",
        "failed": "failed",
    }

    return TaskStatusResponse(
        task_id=rec.processing_task_id or "",
        status=status_map.get(rec.status, rec.status),
        progress=progress,
        result=task_result,
        error=task_error,
    )


@router.get("/videos/{recording_id}/result", response_model=VideoProcessingResult)
async def get_processing_result(
    recording_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """Get full processing results for a completed video."""
    rec = await RecordingRepository.get(db, recording_id)
    if not rec:
        raise HTTPException(404, f"Recording '{recording_id}' not found")
    if rec.status != "completed":
        raise HTTPException(409, f"Recording not completed yet. Status: {rec.status}")

    return VideoProcessingResult(
        recording_id=recording_id,
        status=rec.status,
        total_frames_processed=0,   # TODO: fetch from DetectionSession
        total_persons_detected=0,
        total_faces_detected=0,
        unique_persons_count=0,
        duration_seconds=rec.duration_seconds or 0,
        compression_ratio=rec.compression_ratio,
        summary=[],
    )


@router.get("/videos/{recording_id}/download")
async def get_download_url(
    recording_id: str,
    version: str = "compressed",
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Get presigned download URL (valid 1 hour).

    **version**: `original` | `compressed`
    """
    rec = await RecordingRepository.get(db, recording_id)
    if not rec:
        raise HTTPException(404, f"Recording '{recording_id}' not found")

    storage = get_storage()
    bucket = "videos" if version == "original" else "compressed"
    key = (StorageKeys.video_original(recording_id)
           if version == "original"
           else StorageKeys.video_compressed(recording_id))

    if not await storage.exists(bucket, key):
        raise HTTPException(404, f"{version.capitalize()} file not available")

    url = await storage.get_presigned_url(bucket, key, expires_in=3600)
    return {"recording_id": recording_id, "version": version,
            "url": url, "expires_in_seconds": 3600}


@router.delete("/videos/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(
    recording_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """Delete recording and all associated files from storage and DB."""
    rec = await RecordingRepository.get(db, recording_id)
    if not rec:
        raise HTTPException(404, f"Recording '{recording_id}' not found")

    storage = get_storage()
    for bucket, key in [
        ("videos", StorageKeys.video_original(recording_id)),
        ("compressed", StorageKeys.video_compressed(recording_id)),
    ]:
        try:
            if await storage.exists(bucket, key):
                await storage.delete(bucket, key)
        except Exception:
            pass  # best-effort cleanup

    await RecordingRepository.delete(db, recording_id)


@router.get("/videos", response_model=list[dict])
async def list_recordings(
    camera_id: Optional[str] = None,
    rec_status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """List video recordings with optional filters."""
    recs = await RecordingRepository.list(
        db, camera_id=camera_id, status=rec_status, limit=limit, offset=offset
    )
    return [
        {
            "id": r.id, "camera_id": r.camera_id, "status": r.status,
            "duration_seconds": r.duration_seconds,
            "original_size_mb": round(r.original_size_bytes / 1024 / 1024, 1) if r.original_size_bytes else None,
            "compressed_size_mb": round(r.compressed_size_bytes / 1024 / 1024, 1) if r.compressed_size_bytes else None,
            "compression_ratio": r.compression_ratio,
            "codec_compressed": r.codec_compressed,
            "created_at": r.created_at,
        }
        for r in recs
    ]
