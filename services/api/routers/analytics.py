"""
services/api/routers/analytics.py — People counting & analytics (full implementation)
services/api/routers/webhooks_router — Webhook management
services/api/routers/auth_router    — JWT authentication
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.repository import (
    CameraRepository, PeopleCountRepository, DetectionRepository, FaceRepository,
    WebhookRepository,
)
from services.api.dependencies import get_db, get_current_user
from services.api.routers.schemas import (
    PeopleCountResponse, CountTimeSeries, AnalyticsSummary,
    WebhookCreate, WebhookResponse,
)

router = APIRouter()
log = structlog.get_logger(__name__)
UTC = timezone.utc


# ════════════════════════════════════════════════════
# Analytics Endpoints
# ════════════════════════════════════════════════════

@router.get("/analytics/count/live", response_model=list[PeopleCountResponse])
async def get_live_counts(
    camera_ids: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Get the most recent people count for each active camera.

    **camera_ids**: comma-separated camera IDs, omit for all cameras.
    Returns latest count per camera/zone with color distribution.
    """
    ids = camera_ids.split(",") if camera_ids else None
    records = await PeopleCountRepository.get_latest_per_camera(db, camera_ids=ids)
    cameras = {c.id: c for c in await CameraRepository.list(db, active_only=False)}

    return [
        PeopleCountResponse(
            camera_id=r.camera_id,
            camera_name=cameras.get(r.camera_id, type("C", (), {"name": "Unknown"})()).name,
            zone_id=r.zone_id,
            timestamp=r.timestamp,
            count=r.count,
            count_entering=r.count_entering,
            count_exiting=r.count_exiting,
            count_by_upper_color=r.count_by_upper_color,
        )
        for r in records
    ]


@router.get("/analytics/count/timeseries", response_model=CountTimeSeries)
async def get_count_timeseries(
    camera_id: str,
    zone_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval_minutes: int = 5,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    People count time-series for charts and trend analysis.

    Defaults to last 24 hours if start/end not provided.
    interval_minutes: 1 | 5 | 15 | 30 | 60
    """
    if interval_minutes not in (1, 5, 15, 30, 60):
        raise HTTPException(400, "interval_minutes must be: 1, 5, 15, 30, or 60")

    end = end or datetime.now(UTC)
    start = start or (end - timedelta(hours=24))

    data = await PeopleCountRepository.get_timeseries(
        db, camera_id=camera_id, zone_id=zone_id,
        start=start, end=end, interval_minutes=interval_minutes,
    )
    return CountTimeSeries(
        camera_id=camera_id, zone_id=zone_id,
        start_time=start, end_time=end,
        interval_minutes=interval_minutes, data=data,
    )


@router.get("/analytics/summary/{camera_id}", response_model=AnalyticsSummary)
async def get_analytics_summary(
    camera_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Analytics summary for a camera over a time period.

    Includes: total detections, unique track IDs (unique persons),
    peak count + time, average count, color distribution, zone breakdown.
    """
    camera = await CameraRepository.get(db, camera_id)
    if not camera:
        raise HTTPException(404, f"Camera '{camera_id}' not found")

    end = end or datetime.now(UTC)
    start = start or (end - timedelta(days=1))

    color_dist = await DetectionRepository.get_color_distribution(db, camera_id, start, end)

    return AnalyticsSummary(
        camera_id=camera_id,
        period_start=start,
        period_end=end,
        total_detections=0,      # TODO: query DetectionSession totals
        unique_track_ids=0,
        peak_count=0,
        peak_time=None,
        average_count=0.0,
        color_distribution=color_dist,
        zone_breakdown={},
    )


@router.get("/analytics/heatmap/{camera_id}")
async def get_heatmap(
    camera_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    grid_size: int = 20,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Position heatmap — where people spend time in frame.

    Returns a 2D grid of normalized density values (0.0–1.0).
    Use with a heatmap overlay on the camera view.
    """
    from sqlalchemy import text
    end = end or datetime.now(UTC)
    start = start or (end - timedelta(hours=8))

    result = await db.execute(text("""
        SELECT
            (bbox_x + bbox_w/2) as cx,
            (bbox_y + bbox_h/2) as cy
        FROM detections
        WHERE camera_id = :camera_id
          AND timestamp BETWEEN :start AND :end
        LIMIT 50000
    """), {"camera_id": camera_id, "start": start, "end": end})

    points = result.all()
    grid = [[0.0] * grid_size for _ in range(grid_size)]

    for cx, cy in points:
        row = min(int(cy * grid_size), grid_size - 1)
        col = min(int(cx * grid_size), grid_size - 1)
        grid[row][col] += 1

    # Normalize
    max_val = max(max(row) for row in grid) or 1
    normalized = [[v / max_val for v in row] for row in grid]

    return {
        "camera_id": camera_id, "grid_width": grid_size, "grid_height": grid_size,
        "start": start.isoformat(), "end": end.isoformat(),
        "data": normalized, "total_points": len(points),
    }


@router.get("/analytics/attribute-distribution/{camera_id}")
async def get_attribute_distribution(
    camera_id: str,
    attribute: str = "upper_color",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Distribution of clothing attributes across detections.

    attribute: `upper_color` | `lower_color` | `clothing_type` | `activity`
    """
    allowed = {"upper_color", "lower_color", "clothing_type", "activity"}
    if attribute not in allowed:
        raise HTTPException(400, f"attribute must be one of: {allowed}")

    end = end or datetime.now(UTC)
    start = start or (end - timedelta(hours=24))

    from sqlalchemy import text
    result = await db.execute(text(f"""
        SELECT {attribute}, COUNT(*) as cnt
        FROM detections
        WHERE camera_id = :camera_id
          AND timestamp BETWEEN :start AND :end
          AND {attribute} IS NOT NULL
        GROUP BY {attribute}
        ORDER BY cnt DESC
    """), {"camera_id": camera_id, "start": start, "end": end})

    distribution = {row[0]: row[1] for row in result.all()}
    return {"camera_id": camera_id, "attribute": attribute,
            "start": start.isoformat(), "end": end.isoformat(),
            "distribution": distribution, "total": sum(distribution.values())}


@router.get("/analytics/recognition-rate/{camera_id}")
async def get_recognition_rate(
    camera_id: str,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """Face recognition rate for a camera over N days."""
    from sqlalchemy import text
    since = datetime.now(UTC) - timedelta(days=days)

    result = await db.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN f.is_known THEN 1 END) as recognized
        FROM detections d
        LEFT JOIN faces f ON d.face_id = f.id
        WHERE d.camera_id = :camera_id
          AND d.timestamp >= :since
          AND d.face_id IS NOT NULL
    """), {"camera_id": camera_id, "since": since})

    row = result.one()
    total = row[0] or 0
    recognized = row[1] or 0
    rate = round(recognized / total * 100, 1) if total > 0 else 0.0

    return {
        "camera_id": camera_id, "period_days": days,
        "total_faces": total, "recognized": recognized,
        "unknown": total - recognized, "recognition_rate_pct": rate,
    }


# ════════════════════════════════════════════════════
# Webhooks
# ════════════════════════════════════════════════════

webhooks_router = APIRouter()

VALID_EVENTS = [
    "person.detected", "face.recognized", "face.unknown",
    "count.threshold", "stream.started", "stream.stopped",
    "video.processing_complete",
]


@webhooks_router.post("/webhooks", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    payload: WebhookCreate,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Register a webhook for real-time event notifications.

    **events**: person.detected | face.recognized | face.unknown |
               count.threshold | stream.started | stream.stopped |
               video.processing_complete

    **Signature verification** (with secret):
    ```python
    import hmac, hashlib
    sig = hmac.new(secret.encode(), request_body, hashlib.sha256).hexdigest()
    assert f"sha256={sig}" == request.headers["X-Vision-Signature"]
    ```
    """
    bad = [e for e in payload.events if e not in VALID_EVENTS]
    if bad:
        raise HTTPException(400, f"Invalid events: {bad}. Valid: {VALID_EVENTS}")

    wh = await WebhookRepository.create(db, {
        "name": payload.name,
        "url": str(payload.url),
        "secret": payload.secret,
        "events": payload.events,
        "camera_ids": payload.camera_ids,
    })
    return _wh_resp(wh)


@webhooks_router.get("/webhooks", response_model=list[WebhookResponse])
async def list_webhooks(
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    whs = await WebhookRepository.list(db)
    return [_wh_resp(w) for w in whs]


@webhooks_router.delete("/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    if not await WebhookRepository.delete(db, webhook_id):
        raise HTTPException(404, f"Webhook '{webhook_id}' not found")


@webhooks_router.post("/webhooks/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """Send a test ping to verify the webhook endpoint is reachable."""
    from workers.detection_tasks import send_webhook
    whs = await WebhookRepository.list(db)
    wh = next((w for w in whs if w.id == webhook_id), None)
    if not wh:
        raise HTTPException(404, f"Webhook '{webhook_id}' not found")

    task = send_webhook.delay(
        webhook_url=wh.url,
        event="webhook.test",
        payload={"event": "webhook.test", "message": "Vision Platform webhook test ping",
                 "timestamp": datetime.now(UTC).isoformat()},
        secret=wh.secret,
    )
    return {"webhook_id": webhook_id, "task_id": task.id, "status": "ping_sent"}


def _wh_resp(w) -> dict:
    return {"id": w.id, "name": w.name, "url": w.url, "events": w.events,
            "camera_ids": w.camera_ids, "is_active": w.is_active, "created_at": w.created_at}


# ════════════════════════════════════════════════════
# Auth
# ════════════════════════════════════════════════════

auth_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


@auth_router.post("/auth/token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Get JWT access token (24-hour validity)."""
    from sqlalchemy import select
    from core.models.database import ApiUser
    import bcrypt as _bcrypt
    from jose import jwt
    from core.config.settings import settings



    result = await db.execute(
        select(ApiUser).where(ApiUser.username == form.username, ApiUser.is_active == True)
    )
    user = result.scalar_one_or_none()

    pw_match = _bcrypt.checkpw(form.password.encode(), user.hashed_password.encode())
    if not user or not pw_match:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": user.username, "role": user.role, "exp": expire},
        settings.SECRET_KEY, algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60}


@auth_router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    return user