"""
services/api/routers/streams.py
Camera management & real-time stream control — full implementation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.repository import CameraRepository, SessionRepository
from services.api.dependencies import get_db, get_current_user
from services.api.routers.schemas import (
    CameraCreate, CameraUpdate, CameraResponse, StreamStartRequest, StreamStatus,
)

router = APIRouter()
log = structlog.get_logger(__name__)
UTC = timezone.utc

# In-memory stream state (use Redis for multi-replica deployments)
_active_streams: dict[str, dict] = {}


@router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    payload: CameraCreate,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """Register a new camera. source_type: rtsp | m3u8 | file | webcam"""
    camera = await CameraRepository.create(db, payload.model_dump())
    return _cam_resp(camera)


@router.get("/cameras", response_model=list[CameraResponse])
async def list_cameras(
    active_only: bool = True, limit: int = 100, offset: int = 0,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    cameras = await CameraRepository.list(db, active_only=active_only, limit=limit, offset=offset)
    return [_cam_resp(c) for c in cameras]


@router.get("/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: str, db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    camera = await CameraRepository.get(db, camera_id)
    if not camera:
        raise HTTPException(404, f"Camera '{camera_id}' not found")
    return _cam_resp(camera)


@router.patch("/cameras/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str, payload: CameraUpdate,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    if not await CameraRepository.get(db, camera_id):
        raise HTTPException(404, f"Camera '{camera_id}' not found")
    updated = await CameraRepository.update(db, camera_id, payload.model_dump(exclude_none=True))
    return _cam_resp(updated)


@router.delete("/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: str, db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    if camera_id in _active_streams:
        await _stop_internal(camera_id, db)
    if not await CameraRepository.delete(db, camera_id):
        raise HTTPException(404, f"Camera '{camera_id}' not found")


@router.post("/streams/{camera_id}/start", response_model=StreamStatus)
async def start_stream(
    camera_id: str, config: StreamStartRequest, background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """
    Start real-time AI analysis of a camera stream.

    filter_criteria — count only matching people:
      {"upper_color": "blue"}           → blue shirts only
      {"activity": "running"}           → running only
      {"upper_color": "red", "activity": "standing"}
    """
    camera = await CameraRepository.get(db, camera_id)
    if not camera:
        raise HTTPException(404, f"Camera '{camera_id}' not found")
    if not camera.is_active:
        raise HTTPException(400, "Camera is inactive")

    if camera_id in _active_streams:
        s = _active_streams[camera_id]
        return StreamStatus(camera_id=camera_id, is_running=True,
                            frames_processed=s.get("frames_processed", 0),
                            current_count=s.get("current_count", 0),
                            started_at=s.get("started_at"))

    session = await SessionRepository.create(db, camera_id)
    started_at = datetime.now(UTC)
    _active_streams[camera_id] = {
        "session_id": session.id, "started_at": started_at,
        "frames_processed": 0, "current_count": 0,
    }

    background_tasks.add_task(
        _stream_background, camera_id=camera_id, session_id=session.id,
        source_url=camera.source_url, source_type=camera.source_type,
        sample_rate=camera.frame_sample_rate, zones=camera.zones,
        stream_config=config.model_dump(),
    )

    log.info("stream.started", camera_id=camera_id, session_id=session.id)
    return StreamStatus(camera_id=camera_id, is_running=True,
                        frames_processed=0, current_count=0, started_at=started_at)


@router.post("/streams/{camera_id}/stop")
async def stop_stream(
    camera_id: str, db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    if camera_id not in _active_streams:
        raise HTTPException(404, "No active stream for this camera")
    await _stop_internal(camera_id, db)
    return {"camera_id": camera_id, "is_running": False}


@router.get("/streams/{camera_id}/status", response_model=StreamStatus)
async def get_stream_status(camera_id: str, _user: dict = Depends(get_current_user)):
    s = _active_streams.get(camera_id)
    if not s:
        return StreamStatus(camera_id=camera_id, is_running=False,
                            frames_processed=0, current_count=0, started_at=None)
    return StreamStatus(camera_id=camera_id, is_running=True,
                        frames_processed=s.get("frames_processed", 0),
                        current_count=s.get("current_count", 0),
                        started_at=s.get("started_at"))


@router.get("/streams", response_model=list[StreamStatus])
async def list_active_streams(_user: dict = Depends(get_current_user)):
    return [StreamStatus(camera_id=cid, is_running=True,
                         frames_processed=s.get("frames_processed", 0),
                         current_count=s.get("current_count", 0),
                         started_at=s.get("started_at"))
            for cid, s in _active_streams.items()]


# ─── Internal helpers ──────────────────────────────────────────

async def _stop_internal(camera_id: str, db: AsyncSession) -> None:
    s = _active_streams.pop(camera_id, None)
    if s and s.get("session_id"):
        await SessionRepository.stop(db, s["session_id"])
    log.info("stream.stopped", camera_id=camera_id)


async def _stream_background(
    camera_id: str, session_id: str, source_url: str, source_type: str,
    sample_rate: int, zones: Optional[dict], stream_config: dict,
) -> None:
    """Background loop: read frames → dispatch Celery tasks."""
    import base64, cv2
    from services.video_processor.stream_reader import StreamConfig, StreamReader, SourceType
    from workers.detection_tasks import process_stream_frame

    try:
        stype = SourceType(source_type)
    except ValueError:
        stype = SourceType.RTSP

    cfg = StreamConfig(source_id=camera_id, source_url=source_url,
                       source_type=stype, sample_rate=sample_rate)
    reader = StreamReader(cfg)

    try:
        async for frame in reader.frames():
            if camera_id not in _active_streams:
                break

            _, buf = cv2.imencode(".jpg", frame.data, [cv2.IMWRITE_JPEG_QUALITY, 80])
            payload = {
                "frame_b64": base64.b64encode(buf.tobytes()).decode(),
                "camera_id": camera_id, "session_id": session_id,
                "frame_number": frame.frame_number, "timestamp": frame.timestamp,
                "zones": zones, **stream_config,
            }
            process_stream_frame.apply_async(args=[payload], queue="gpu_tasks", expires=30)
            if camera_id in _active_streams:
                _active_streams[camera_id]["frames_processed"] = frame.frame_number
    except Exception as e:
        log.error("stream.background_error", camera_id=camera_id, error=str(e))
    finally:
        reader.stop()
        _active_streams.pop(camera_id, None)


def _cam_resp(c) -> dict:
    return {"id": c.id, "name": c.name, "description": c.description,
            "location": c.location, "source_type": c.source_type,
            "source_url": c.source_url, "fps_target": c.fps_target,
            "is_active": c.is_active, "is_recording": c.is_recording,
            "last_seen_at": c.last_seen_at, "zones": c.zones, "created_at": c.created_at}
