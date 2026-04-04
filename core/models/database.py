"""
core/models/database.py
SQLAlchemy ORM models for PostgreSQL.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def new_uuid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────
# Camera / Source
# ─────────────────────────────────────────────────────────────
class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    location: Mapped[Optional[str]] = mapped_column(String(255))  # "Lobby A", "Gate 1"
    source_type: Mapped[str] = mapped_column(String(20))  # "rtsp" | "m3u8" | "file" | "webcam"
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Stream config
    fps_target: Mapped[int] = mapped_column(Integer, default=5)    # frames/sec to process
    frame_sample_rate: Mapped[int] = mapped_column(Integer, default=5)
    resolution_width: Mapped[Optional[int]] = mapped_column(Integer)
    resolution_height: Mapped[Optional[int]] = mapped_column(Integer)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_recording: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Zone config (JSON): defines counting zones as polygons
    zones: Mapped[Optional[dict]] = mapped_column(JSON)
    # Example: {"zone_a": [[x1,y1],[x2,y2],...], "entrance": [[...]]}

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    detection_sessions: Mapped[list["DetectionSession"]] = relationship(back_populates="camera")
    video_recordings: Mapped[list["VideoRecording"]] = relationship(back_populates="camera")


# ─────────────────────────────────────────────────────────────
# Detection Session (one session = one stream analysis run)
# ─────────────────────────────────────────────────────────────
class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    status: Mapped[str] = mapped_column(String(20), default="running")
    # "running" | "completed" | "failed" | "stopped"

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    total_frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    total_faces_detected: Mapped[int] = mapped_column(Integer, default=0)
    total_faces_recognized: Mapped[int] = mapped_column(Integer, default=0)

    camera: Mapped["Camera"] = relationship(back_populates="detection_sessions")
    detections: Mapped[list["Detection"]] = relationship(back_populates="session")
    people_counts: Mapped[list["PeopleCount"]] = relationship(back_populates="session")


# ─────────────────────────────────────────────────────────────
# Detection (one detection = one person in one frame)
# ─────────────────────────────────────────────────────────────
class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("detection_sessions.id"), index=True)
    camera_id: Mapped[str] = mapped_column(String(36), index=True)

    # Frame info
    frame_number: Mapped[int] = mapped_column(BigInteger)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    frame_path: Mapped[Optional[str]] = mapped_column(Text)  # MinIO path to frame

    # Bounding box (normalized 0-1)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)

    # Detection info
    track_id: Mapped[Optional[int]] = mapped_column(Integer)   # ByteTrack ID
    confidence: Mapped[float] = mapped_column(Float)
    class_name: Mapped[str] = mapped_column(String(50), default="person")

    # Attribute analysis
    upper_color: Mapped[Optional[str]] = mapped_column(String(50))  # "red", "blue", etc.
    lower_color: Mapped[Optional[str]] = mapped_column(String(50))
    clothing_type: Mapped[Optional[str]] = mapped_column(String(100))  # "shirt", "jacket"
    activity: Mapped[Optional[str]] = mapped_column(String(100))  # "walking", "standing"
    attributes: Mapped[Optional[dict]] = mapped_column(JSON)  # raw attribute dict

    # Zone
    zone_id: Mapped[Optional[str]] = mapped_column(String(100))  # which zone this person is in

    # Face link
    face_id: Mapped[Optional[str]] = mapped_column(ForeignKey("faces.id"), nullable=True)
    face_confidence: Mapped[Optional[float]] = mapped_column(Float)

    session: Mapped["DetectionSession"] = relationship(back_populates="detections")
    face: Mapped[Optional["Face"]] = relationship()


# ─────────────────────────────────────────────────────────────
# Face
# ─────────────────────────────────────────────────────────────
class Face(Base):
    __tablename__ = "faces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)

    # Identity
    person_id: Mapped[Optional[str]] = mapped_column(ForeignKey("persons.id"), nullable=True)
    cluster_id: Mapped[Optional[str]] = mapped_column(String(100))  # DBSCAN cluster label
    is_known: Mapped[bool] = mapped_column(Boolean, default=False)

    # Best capture
    best_frame_path: Mapped[Optional[str]] = mapped_column(Text)  # MinIO path
    capture_count: Mapped[int] = mapped_column(Integer, default=1)
    best_quality_score: Mapped[Optional[float]] = mapped_column(Float)  # sharpness/size

    # Embedding stored in pgvector (also stored in vector_store)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(512))  # for pgvector direct

    # Attributes from InsightFace
    age_estimate: Mapped[Optional[int]] = mapped_column(Integer)
    gender: Mapped[Optional[str]] = mapped_column(String(10))  # "M" | "F"
    detection_score: Mapped[Optional[float]] = mapped_column(Float)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    camera_ids: Mapped[Optional[list]] = mapped_column(JSON)  # cameras this face appeared on

    person: Mapped[Optional["Person"]] = relationship(back_populates="faces")


# ─────────────────────────────────────────────────────────────
# Person (known identity)
# ─────────────────────────────────────────────────────────────
class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255))
    employee_id: Mapped[Optional[str]] = mapped_column(String(100))
    department: Mapped[Optional[str]] = mapped_column(String(100))
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    is_watchlist: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    faces: Mapped[list["Face"]] = relationship(back_populates="person")


# ─────────────────────────────────────────────────────────────
# People Count (time-series counting data)
# ─────────────────────────────────────────────────────────────
class PeopleCount(Base):
    __tablename__ = "people_counts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("detection_sessions.id"), index=True)
    camera_id: Mapped[str] = mapped_column(String(36), index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(String(100))

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    count_entering: Mapped[int] = mapped_column(Integer, default=0)
    count_exiting: Mapped[int] = mapped_column(Integer, default=0)

    # Attribute breakdown
    count_by_upper_color: Mapped[Optional[dict]] = mapped_column(JSON)
    # Example: {"red": 3, "blue": 2, "white": 5}

    session: Mapped["DetectionSession"] = relationship(back_populates="people_counts")


# ─────────────────────────────────────────────────────────────
# Video Recording
# ─────────────────────────────────────────────────────────────
class VideoRecording(Base):
    __tablename__ = "video_recordings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(20))  # "upload" | "stream_capture"

    # Files
    original_path: Mapped[Optional[str]] = mapped_column(Text)  # MinIO path
    compressed_path: Mapped[Optional[str]] = mapped_column(Text)  # MinIO path

    # Metadata
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    original_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    compressed_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    compression_ratio: Mapped[Optional[float]] = mapped_column(Float)

    codec_original: Mapped[Optional[str]] = mapped_column(String(50))
    codec_compressed: Mapped[Optional[str]] = mapped_column(String(50))
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    fps: Mapped[Optional[float]] = mapped_column(Float)

    # Processing status
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    # "uploaded" | "queued" | "processing" | "completed" | "failed"
    processing_task_id: Mapped[Optional[str]] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    camera: Mapped[Optional["Camera"]] = relationship(back_populates="video_recordings")


# ─────────────────────────────────────────────────────────────
# Webhook
# ─────────────────────────────────────────────────────────────
class Webhook(Base):
    __tablename__ = "webhooks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(Text)
    secret: Mapped[Optional[str]] = mapped_column(String(255))

    # Events to trigger on
    events: Mapped[list] = mapped_column(JSON)
    # ["person.detected", "face.recognized", "count.threshold", "face.unknown"]

    camera_ids: Mapped[Optional[list]] = mapped_column(JSON)  # null = all cameras

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ─────────────────────────────────────────────────────────────
# API User (authentication)
# ─────────────────────────────────────────────────────────────
class ApiUser(Base):
    __tablename__ = "api_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    # "admin" | "operator" | "viewer"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
