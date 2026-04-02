"""
services/api/routers/streams.py — Camera & Stream management
services/api/routers/videos.py  — Video upload & processing
services/api/routers/faces.py   — Face enrollment & recognition
services/api/routers/analytics.py — People counting & analytics
"""
from __future__ import annotations

# ════════════════════════════════════════════════════════
# SHARED PYDANTIC MODELS (request/response schemas)
# ════════════════════════════════════════════════════════

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl


# ── Camera / Stream ─────────────────────────────────────
class CameraCreate(BaseModel):
    name: str = Field(..., example="Lobby Entrance A")
    description: Optional[str] = Field(None, example="Main lobby entrance facing north")
    location: Optional[str] = Field(None, example="Building A - Floor 1")
    source_type: str = Field(..., example="rtsp")  # rtsp | m3u8 | file | webcam
    source_url: str = Field(..., example="rtsp://192.168.1.100:554/stream1")
    fps_target: int = Field(5, ge=1, le=30, example=5)
    frame_sample_rate: int = Field(5, ge=1, le=30, example=5)
    zones: Optional[dict[str, list[list[float]]]] = Field(
        None,
        example={
            "entrance": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "exit": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        },
        description="Named polygon zones for counting. Coordinates are normalized 0.0-1.0."
    )


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    fps_target: Optional[int] = None
    zones: Optional[dict] = None
    is_active: Optional[bool] = None


class CameraResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    location: Optional[str]
    source_type: str
    source_url: str
    fps_target: int
    is_active: bool
    is_recording: bool
    last_seen_at: Optional[datetime]
    zones: Optional[dict]
    created_at: datetime


class StreamStartRequest(BaseModel):
    camera_id: str
    extract_faces: bool = Field(True, description="Run face recognition on detected persons")
    analyze_attributes: bool = Field(True, description="Analyze clothing color and attributes")
    filter_criteria: Optional[dict[str, str]] = Field(
        None,
        example={"upper_color": "blue"},
        description="Only count people matching these attributes. Null = count all."
    )


class StreamStatus(BaseModel):
    camera_id: str
    is_running: bool
    frames_processed: int
    current_count: int
    started_at: Optional[datetime]


# ── Video Processing ─────────────────────────────────────
class VideoUploadResponse(BaseModel):
    recording_id: str
    task_id: str
    status: str = "queued"
    message: str


class VideoProcessingConfig(BaseModel):
    extract_faces: bool = True
    analyze_attributes: bool = True
    compress: bool = True
    sample_rate: int = Field(5, ge=1, le=30)
    zones: Optional[dict] = None
    filter_criteria: Optional[dict[str, str]] = None


class VideoProcessingResult(BaseModel):
    recording_id: str
    status: str
    total_frames_processed: int
    total_persons_detected: int
    total_faces_detected: int
    unique_persons_count: int
    duration_seconds: float
    compression_ratio: Optional[float]
    summary: list[dict]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str    # "pending" | "running" | "completed" | "failed"
    progress: Optional[float]
    result: Optional[dict]
    error: Optional[str]


# ── Face Recognition ─────────────────────────────────────
class PersonEnrollRequest(BaseModel):
    name: str = Field(..., example="John Doe")
    employee_id: Optional[str] = Field(None, example="EMP001")
    department: Optional[str] = Field(None, example="Engineering")
    metadata: Optional[dict] = None
    is_watchlist: bool = False


class PersonResponse(BaseModel):
    id: str
    name: str
    employee_id: Optional[str]
    department: Optional[str]
    face_count: int
    is_watchlist: bool
    created_at: datetime


class FaceResponse(BaseModel):
    id: str
    person_id: Optional[str]
    person_name: Optional[str]
    cluster_id: Optional[str]
    is_known: bool
    best_frame_url: Optional[str]  # presigned MinIO URL
    capture_count: int
    quality_score: Optional[float]
    age_estimate: Optional[int]
    gender: Optional[str]
    first_seen_at: datetime
    last_seen_at: datetime
    camera_ids: Optional[list[str]]


class FaceSearchRequest(BaseModel):
    threshold: float = Field(0.45, ge=0.0, le=1.0,
                              description="Similarity threshold. Higher = more strict match.")
    top_k: int = Field(5, ge=1, le=20)
    camera_ids: Optional[list[str]] = None


class FaceSearchResult(BaseModel):
    face_id: str
    person_id: Optional[str]
    person_name: Optional[str]
    similarity_score: float
    best_frame_url: Optional[str]


class FaceClusterResponse(BaseModel):
    cluster_id: str
    face_count: int
    representative_face_url: Optional[str]
    first_seen: datetime
    last_seen: datetime
    camera_ids: list[str]


# ── Analytics ────────────────────────────────────────────
class PeopleCountResponse(BaseModel):
    camera_id: str
    camera_name: str
    zone_id: Optional[str]
    timestamp: datetime
    count: int
    count_entering: int
    count_exiting: int
    count_by_upper_color: Optional[dict[str, int]]


class CountTimeSeries(BaseModel):
    camera_id: str
    zone_id: Optional[str]
    start_time: datetime
    end_time: datetime
    interval_minutes: int
    data: list[dict[str, Any]]  # [{timestamp, count, entering, exiting}, ...]


class AnalyticsSummary(BaseModel):
    camera_id: str
    period_start: datetime
    period_end: datetime
    total_detections: int
    unique_track_ids: int
    peak_count: int
    peak_time: Optional[datetime]
    average_count: float
    color_distribution: dict[str, int]  # {"blue": 45, "white": 30, ...}
    zone_breakdown: dict[str, int]


# ── Webhooks ─────────────────────────────────────────────
class WebhookCreate(BaseModel):
    name: str = Field(..., example="Alert System")
    url: str = Field(..., example="https://your-app.com/webhook/vision")
    secret: Optional[str] = Field(None, description="HMAC secret for signature verification")
    events: list[str] = Field(
        ...,
        example=["person.detected", "face.recognized", "face.unknown", "count.threshold"],
        description="Events to subscribe to"
    )
    camera_ids: Optional[list[str]] = Field(
        None, description="Specific cameras to watch. Null = all cameras."
    )


class WebhookResponse(BaseModel):
    id: str
    name: str
    url: str
    events: list[str]
    camera_ids: Optional[list[str]]
    is_active: bool
    created_at: datetime


# ════════════════════════════════════════════════════════
# WEBHOOK PAYLOAD SCHEMAS
# These are the bodies sent TO your webhook endpoint
# ════════════════════════════════════════════════════════

class WebhookPayloadPersonDetected(BaseModel):
    """Fired when a person is detected in a frame."""
    event: str = "person.detected"
    timestamp: str
    camera_id: str
    camera_name: str
    frame_number: int
    detections: list[dict]  # list of person detection objects
    total_count: int
    zone_counts: dict[str, int]


class WebhookPayloadFaceRecognized(BaseModel):
    """Fired when a known face is recognized."""
    event: str = "face.recognized"
    timestamp: str
    camera_id: str
    person_id: str
    person_name: str
    similarity_score: float
    face_crop_url: Optional[str]  # temporary presigned URL


class WebhookPayloadFaceUnknown(BaseModel):
    """Fired when an unknown face is detected."""
    event: str = "face.unknown"
    timestamp: str
    camera_id: str
    face_id: str  # temporary ID for this unrecognized face
    face_crop_url: Optional[str]


class WebhookPayloadCountThreshold(BaseModel):
    """Fired when count exceeds/drops below a threshold."""
    event: str = "count.threshold"
    timestamp: str
    camera_id: str
    zone_id: Optional[str]
    current_count: int
    threshold: int
    direction: str  # "above" | "below"
