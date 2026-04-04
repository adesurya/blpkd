"""
core/models/repository.py

Database repository pattern — semua query DB terpusat di sini.
Setiap router hanya memanggil repository, tidak langsung query DB.
Ini memudahkan testing dan maintenance.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

import structlog
from sqlalchemy import func, select, update, delete, desc, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.database import (
    Camera, DetectionSession, Detection, Face, Person,
    PeopleCount, VideoRecording, Webhook, ApiUser,
)

log = structlog.get_logger(__name__)

UTC = timezone.utc


def new_id() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────
# Camera Repository
# ─────────────────────────────────────────────────────────────
class CameraRepository:

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> Camera:
        cam = Camera(id=new_id(), **data)
        db.add(cam)
        await db.flush()
        await db.refresh(cam)
        log.info("camera.created", id=cam.id, name=cam.name)
        return cam

    @staticmethod
    async def get(db: AsyncSession, camera_id: str) -> Optional[Camera]:
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Camera]:
        q = select(Camera)
        if active_only:
            q = q.where(Camera.is_active == True)
        q = q.order_by(Camera.created_at.desc()).limit(limit).offset(offset)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def update(db: AsyncSession, camera_id: str, data: dict) -> Optional[Camera]:
        data = {k: v for k, v in data.items() if v is not None}
        data["updated_at"] = datetime.now(UTC)
        await db.execute(
            update(Camera).where(Camera.id == camera_id).values(**data)
        )
        return await CameraRepository.get(db, camera_id)

    @staticmethod
    async def delete(db: AsyncSession, camera_id: str) -> bool:
        result = await db.execute(delete(Camera).where(Camera.id == camera_id))
        return result.rowcount > 0

    @staticmethod
    async def touch(db: AsyncSession, camera_id: str) -> None:
        """Update last_seen_at timestamp."""
        await db.execute(
            update(Camera)
            .where(Camera.id == camera_id)
            .values(last_seen_at=datetime.now(UTC))
        )


# ─────────────────────────────────────────────────────────────
# Detection Session Repository
# ─────────────────────────────────────────────────────────────
class SessionRepository:

    @staticmethod
    async def create(db: AsyncSession, camera_id: str) -> DetectionSession:
        session = DetectionSession(id=new_id(), camera_id=camera_id)
        db.add(session)
        await db.flush()
        await db.refresh(session)
        return session

    @staticmethod
    async def get(db: AsyncSession, session_id: str) -> Optional[DetectionSession]:
        result = await db.execute(
            select(DetectionSession).where(DetectionSession.id == session_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_active_for_camera(
        db: AsyncSession, camera_id: str
    ) -> Optional[DetectionSession]:
        result = await db.execute(
            select(DetectionSession).where(
                and_(
                    DetectionSession.camera_id == camera_id,
                    DetectionSession.status == "running",
                )
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def stop(db: AsyncSession, session_id: str) -> None:
        await db.execute(
            update(DetectionSession)
            .where(DetectionSession.id == session_id)
            .values(status="stopped", ended_at=datetime.now(UTC))
        )

    @staticmethod
    async def increment_counters(
        db: AsyncSession,
        session_id: str,
        frames: int = 0,
        detections: int = 0,
        faces_detected: int = 0,
        faces_recognized: int = 0,
    ) -> None:
        await db.execute(
            update(DetectionSession)
            .where(DetectionSession.id == session_id)
            .values(
                total_frames_processed=DetectionSession.total_frames_processed + frames,
                total_detections=DetectionSession.total_detections + detections,
                total_faces_detected=DetectionSession.total_faces_detected + faces_detected,
                total_faces_recognized=DetectionSession.total_faces_recognized + faces_recognized,
            )
        )


# ─────────────────────────────────────────────────────────────
# Detection Repository
# ─────────────────────────────────────────────────────────────
class DetectionRepository:

    @staticmethod
    async def bulk_insert(db: AsyncSession, detections: list[dict]) -> None:
        """Bulk insert detections for efficiency."""
        if not detections:
            return
        objects = [Detection(id=new_id(), **d) for d in detections]
        db.add_all(objects)
        await db.flush()

    @staticmethod
    async def get_latest_count(
        db: AsyncSession,
        camera_id: str,
        zone_id: Optional[str] = None,
        since_seconds: int = 10,
    ) -> int:
        """Get current person count for a camera (last N seconds)."""
        since = datetime.now(UTC) - timedelta(seconds=since_seconds)
        q = select(func.count(Detection.id.distinct())).where(
            and_(
                Detection.camera_id == camera_id,
                Detection.timestamp >= since,
            )
        )
        if zone_id:
            q = q.where(Detection.zone_id == zone_id)
        result = await db.execute(q)
        return result.scalar() or 0

    @staticmethod
    async def get_color_distribution(
        db: AsyncSession,
        camera_id: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, int]:
        result = await db.execute(
            select(Detection.upper_color, func.count(Detection.id))
            .where(
                and_(
                    Detection.camera_id == camera_id,
                    Detection.timestamp.between(start, end),
                    Detection.upper_color.isnot(None),
                )
            )
            .group_by(Detection.upper_color)
        )
        return {row[0]: row[1] for row in result.all()}


# ─────────────────────────────────────────────────────────────
# People Count Repository
# ─────────────────────────────────────────────────────────────
class PeopleCountRepository:

    @staticmethod
    async def insert(db: AsyncSession, data: dict) -> PeopleCount:
        record = PeopleCount(id=new_id(), **data)
        db.add(record)
        await db.flush()
        return record

    @staticmethod
    async def get_latest_per_camera(
        db: AsyncSession,
        camera_ids: Optional[list[str]] = None,
    ) -> list[PeopleCount]:
        """Get the most recent count record for each camera/zone pair."""
        # Subquery: max timestamp per camera+zone
        subq = (
            select(
                PeopleCount.camera_id,
                PeopleCount.zone_id,
                func.max(PeopleCount.timestamp).label("max_ts"),
            )
            .group_by(PeopleCount.camera_id, PeopleCount.zone_id)
            .subquery()
        )
        q = select(PeopleCount).join(
            subq,
            and_(
                PeopleCount.camera_id == subq.c.camera_id,
                PeopleCount.timestamp == subq.c.max_ts,
            ),
        )
        if camera_ids:
            q = q.where(PeopleCount.camera_id.in_(camera_ids))
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def get_timeseries(
        db: AsyncSession,
        camera_id: str,
        zone_id: Optional[str],
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> list[dict]:
        """Aggregate count data into time-series buckets."""
        interval_sql = f"{interval_minutes} minutes"
        # asyncpg tidak bisa infer tipe parameter NULL
        # Gunakan query berbeda tergantung apakah zone_id di-set atau tidak
        if zone_id:
            q = text("""
                SELECT
                    date_trunc('minute', timestamp) -
                        (EXTRACT(MINUTE FROM timestamp)::int % :interval * interval '1 minute')
                        AS bucket,
                    AVG(count)::int     AS count,
                    SUM(count_entering) AS entering,
                    SUM(count_exiting)  AS exiting
                FROM people_counts
                WHERE camera_id = :camera_id
                  AND timestamp BETWEEN :start AND :end
                  AND zone_id = :zone_id
                GROUP BY bucket
                ORDER BY bucket
            """)
            result = await db.execute(q, {
                "camera_id": camera_id,
                "start": start,
                "end": end,
                "zone_id": zone_id,
                "interval": interval_minutes,
            })
        else:
            q = text("""
                SELECT
                    date_trunc('minute', timestamp) -
                        (EXTRACT(MINUTE FROM timestamp)::int % :interval * interval '1 minute')
                        AS bucket,
                    AVG(count)::int     AS count,
                    SUM(count_entering) AS entering,
                    SUM(count_exiting)  AS exiting
                FROM people_counts
                WHERE camera_id = :camera_id
                  AND timestamp BETWEEN :start AND :end
                GROUP BY bucket
                ORDER BY bucket
            """)
            result = await db.execute(q, {
                "camera_id": camera_id,
                "start": start,
                "end": end,
                "interval": interval_minutes,
            })
        rows = result.mappings().all()
        return [
            {
                "timestamp": row["bucket"].isoformat(),
                "count": row["count"] or 0,
                "entering": row["entering"] or 0,
                "exiting": row["exiting"] or 0,
            }
            for row in rows
        ]


# ─────────────────────────────────────────────────────────────
# Person Repository
# ─────────────────────────────────────────────────────────────
class PersonRepository:

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> Person:
        person = Person(id=new_id(), **data)
        db.add(person)
        await db.flush()
        await db.refresh(person)
        return person

    @staticmethod
    async def get(db: AsyncSession, person_id: str) -> Optional[Person]:
        result = await db.execute(select(Person).where(Person.id == person_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        watchlist_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Person]:
        q = select(Person)
        if watchlist_only:
            q = q.where(Person.is_watchlist == True)
        q = q.order_by(Person.created_at.desc()).limit(limit).offset(offset)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def delete(db: AsyncSession, person_id: str) -> bool:
        result = await db.execute(delete(Person).where(Person.id == person_id))
        return result.rowcount > 0

    @staticmethod
    async def get_face_count(db: AsyncSession, person_id: str) -> int:
        result = await db.execute(
            select(func.count(Face.id)).where(Face.person_id == person_id)
        )
        return result.scalar() or 0


# ─────────────────────────────────────────────────────────────
# Face Repository
# ─────────────────────────────────────────────────────────────
class FaceRepository:

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> Face:
        face = Face(id=new_id(), **data)
        db.add(face)
        await db.flush()
        await db.refresh(face)
        return face

    @staticmethod
    async def get(db: AsyncSession, face_id: str) -> Optional[Face]:
        result = await db.execute(select(Face).where(Face.id == face_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        camera_id: Optional[str] = None,
        is_known: Optional[bool] = None,
        cluster_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Face]:
        q = select(Face)
        if is_known is not None:
            q = q.where(Face.is_known == is_known)
        if cluster_id:
            q = q.where(Face.cluster_id == cluster_id)
        if camera_id:
            q = q.where(Face.camera_ids.contains([camera_id]))
        q = q.order_by(desc(Face.last_seen_at)).limit(limit).offset(offset)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def get_unknown_with_embeddings(
        db: AsyncSession, limit: int = 5000
    ) -> list[Face]:
        """Get unknown faces that have embeddings for DBSCAN clustering."""
        result = await db.execute(
            select(Face)
            .where(
                and_(Face.is_known == False, Face.embedding.isnot(None))
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_cluster(
        db: AsyncSession, face_ids: list[str], cluster_id: str
    ) -> None:
        await db.execute(
            update(Face)
            .where(Face.id.in_(face_ids))
            .values(cluster_id=cluster_id)
        )

    @staticmethod
    async def assign_person(
        db: AsyncSession, face_ids: list[str], person_id: str
    ) -> int:
        result = await db.execute(
            update(Face)
            .where(Face.id.in_(face_ids))
            .values(person_id=person_id, is_known=True)
        )
        return result.rowcount

    @staticmethod
    async def delete(db: AsyncSession, face_id: str) -> bool:
        result = await db.execute(delete(Face).where(Face.id == face_id))
        return result.rowcount > 0

    @staticmethod
    async def get_clusters_summary(
        db: AsyncSession, min_size: int = 2
    ) -> list[dict]:
        """Get DBSCAN cluster summaries for the cluster list endpoint."""
        result = await db.execute(text("""
            SELECT
                cluster_id,
                COUNT(*) as face_count,
                MIN(first_seen_at) as first_seen,
                MAX(last_seen_at) as last_seen,
                MIN(best_frame_path) as representative_path
            FROM faces
            WHERE cluster_id IS NOT NULL
              AND cluster_id != 'noise'
              AND is_known = false
            GROUP BY cluster_id
            HAVING COUNT(*) >= :min_size
            ORDER BY face_count DESC
        """), {"min_size": min_size})
        return [dict(row._mapping) for row in result.all()]

    @staticmethod
    async def update_last_seen(db: AsyncSession, face_id: str) -> None:
        await db.execute(
            update(Face)
            .where(Face.id == face_id)
            .values(
                last_seen_at=datetime.now(UTC),
                capture_count=Face.capture_count + 1,
            )
        )


# ─────────────────────────────────────────────────────────────
# Video Recording Repository
# ─────────────────────────────────────────────────────────────
class RecordingRepository:

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> VideoRecording:
        rec = VideoRecording(id=new_id(), **data)
        db.add(rec)
        await db.flush()
        await db.refresh(rec)
        return rec

    @staticmethod
    async def get(db: AsyncSession, recording_id: str) -> Optional[VideoRecording]:
        result = await db.execute(
            select(VideoRecording).where(VideoRecording.id == recording_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        db: AsyncSession,
        camera_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[VideoRecording]:
        q = select(VideoRecording)
        if camera_id:
            q = q.where(VideoRecording.camera_id == camera_id)
        if status:
            q = q.where(VideoRecording.status == status)
        q = q.order_by(desc(VideoRecording.created_at)).limit(limit).offset(offset)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def update_status(
        db: AsyncSession,
        recording_id: str,
        status: str,
        extra: Optional[dict] = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(UTC),
        }
        if extra:
            values.update(extra)
        await db.execute(
            update(VideoRecording)
            .where(VideoRecording.id == recording_id)
            .values(**values)
        )

    @staticmethod
    async def delete(db: AsyncSession, recording_id: str) -> bool:
        result = await db.execute(
            delete(VideoRecording).where(VideoRecording.id == recording_id)
        )
        return result.rowcount > 0


# ─────────────────────────────────────────────────────────────
# Webhook Repository
# ─────────────────────────────────────────────────────────────
class WebhookRepository:

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> Webhook:
        wh = Webhook(id=new_id(), **data)
        db.add(wh)
        await db.flush()
        await db.refresh(wh)
        return wh

    @staticmethod
    async def list(db: AsyncSession, active_only: bool = True) -> list[Webhook]:
        q = select(Webhook)
        if active_only:
            q = q.where(Webhook.is_active == True)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def get_for_event(
        db: AsyncSession, event: str, camera_id: Optional[str] = None
    ) -> list[Webhook]:
        """Get active webhooks subscribed to a specific event."""
        result = await db.execute(
            select(Webhook).where(
                and_(
                    Webhook.is_active == True,
                    Webhook.events.contains([event]),
                )
            )
        )
        webhooks = list(result.scalars().all())
        if camera_id:
            # Filter: webhook must cover this camera or cover all cameras (null)
            webhooks = [
                w for w in webhooks
                if w.camera_ids is None or camera_id in w.camera_ids
            ]
        return webhooks

    @staticmethod
    async def delete(db: AsyncSession, webhook_id: str) -> bool:
        result = await db.execute(delete(Webhook).where(Webhook.id == webhook_id))
        return result.rowcount > 0