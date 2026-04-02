"""
services/api/routers/faces.py
Face enrollment, recognition, cluster management — full implementation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.abstractions.storage import get_storage, StorageKeys
from core.models.repository import PersonRepository, FaceRepository
from services.api.dependencies import get_db, get_current_user
from services.api.routers.schemas import (
    PersonEnrollRequest, PersonResponse,
    FaceResponse, FaceSearchResult, FaceClusterResponse,
)

router = APIRouter()
log = structlog.get_logger(__name__)
UTC = timezone.utc


# ════════════════════════════════════════════════════
# PERSON / ENROLLMENT
# ════════════════════════════════════════════════════

@router.post("/persons", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
async def create_person(
    payload: PersonEnrollRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """Create a known person record. Enroll face photos with POST /persons/{id}/enroll."""
    person = await PersonRepository.create(db, payload.model_dump())
    return _person_resp(person, face_count=0)


@router.post("/persons/{person_id}/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_face(
    person_id: str,
    photos: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    """
    Enroll 1–20 face photos for a known person.

    Best practices:
    - Front-facing, clear, well-lit photos
    - One face per photo
    - Min 100×100px face resolution
    - 3–10 photos for good accuracy
    """
    import cv2, numpy as np

    person = await PersonRepository.get(db, person_id)
    if not person:
        raise HTTPException(404, f"Person '{person_id}' not found")

    if len(photos) < 1 or len(photos) > 20:
        raise HTTPException(400, "Upload 1–20 photos per enrollment call")

    from services.face_engine.recognizer import FaceEngine
    engine = FaceEngine.get_instance()
    storage = get_storage()

    enrolled_count = 0
    errors: list[str] = []

    for photo in photos:
        data = await photo.read()
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if bgr is None:
            errors.append(f"{photo.filename}: cannot decode image")
            continue

        faces = engine.detect_faces(bgr)
        if not faces:
            errors.append(f"{photo.filename}: no face detected — use clear front-facing photo")
            continue
        if len(faces) > 1:
            errors.append(f"{photo.filename}: {len(faces)} faces detected — use single-face photos")
            continue

        face = faces[0]
        face_id = str(uuid.uuid4())

        # Save face crop to MinIO
        _, buf = cv2.imencode(".jpg", face.crop)
        try:
            await storage.upload(
                bucket="faces",
                key=StorageKeys.face_crop(face_id),
                data=bytes(buf),
                content_type="image/jpeg",
                metadata={"person_id": person_id, "face_id": face_id},
            )
        except Exception as e:
            log.warning("enroll.storage_failed", error=str(e))

        # Register embedding in vector store
        await engine.register_face(
            face, face_id=face_id, person_id=person_id,
            metadata={"person_name": person.name},
        )

        # Save face record to DB
        await FaceRepository.create(db, {
            "id": face_id,
            "person_id": person_id,
            "is_known": True,
            "best_frame_path": StorageKeys.face_crop(face_id),
            "embedding": face.embedding.tolist(),
            "detection_score": face.detection_score,
            "quality_score": face.quality_score,
            "age_estimate": face.age,
            "gender": face.gender,
            "camera_ids": [],
        })

        enrolled_count += 1
        log.info("enroll.face_enrolled", face_id=face_id, person_id=person_id)

    if enrolled_count == 0:
        raise HTTPException(422, {"message": "No faces enrolled", "errors": errors})

    return {
        "person_id": person_id,
        "enrolled_count": enrolled_count,
        "errors": errors,
        "message": f"Enrolled {enrolled_count} face(s) successfully.",
    }


@router.get("/persons", response_model=list[PersonResponse])
async def list_persons(
    watchlist_only: bool = False, limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    persons = await PersonRepository.list(db, watchlist_only=watchlist_only, limit=limit, offset=offset)
    result = []
    for p in persons:
        count = await PersonRepository.get_face_count(db, p.id)
        result.append(_person_resp(p, face_count=count))
    return result


@router.get("/persons/{person_id}", response_model=PersonResponse)
async def get_person(
    person_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    person = await PersonRepository.get(db, person_id)
    if not person:
        raise HTTPException(404, f"Person '{person_id}' not found")
    count = await PersonRepository.get_face_count(db, person_id)
    return _person_resp(person, face_count=count)


@router.delete("/persons/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(
    person_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """Delete person and remove all their face embeddings from vector store."""
    person = await PersonRepository.get(db, person_id)
    if not person:
        raise HTTPException(404, f"Person '{person_id}' not found")

    # Remove all face embeddings from vector store
    from core.abstractions.vector_store import get_vector_store
    faces = await FaceRepository.list(db, is_known=True)
    person_faces = [f for f in faces if f.person_id == person_id]
    store = get_vector_store()
    for face in person_faces:
        try:
            await store.delete("faces", face.id)
        except Exception:
            pass

    await PersonRepository.delete(db, person_id)


# ════════════════════════════════════════════════════
# FACE SEARCH & RECOGNITION
# ════════════════════════════════════════════════════

@router.post("/faces/search", response_model=list[FaceSearchResult])
async def search_face_by_photo(
    photo: UploadFile = File(...),
    threshold: float = 0.45,
    top_k: int = 5,
    _user: dict = Depends(get_current_user),
):
    """
    Search enrolled persons by uploading a face photo.

    threshold 0.45=lenient, 0.60=strict
    Returns top-k matches sorted by similarity (highest first).
    """
    import cv2, numpy as np

    data = await photo.read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Cannot decode uploaded image")

    from services.face_engine.recognizer import FaceEngine
    engine = FaceEngine.get_instance()
    faces = engine.detect_faces(bgr)

    if not faces:
        raise HTTPException(422, "No face detected in uploaded photo")

    results = []
    storage = get_storage()
    for face in faces[:1]:  # search with best quality face
        match = await engine.recognize(face, threshold=threshold)
        if match.is_known:
            face_url = None
            if match.matched_face_id:
                try:
                    face_url = await storage.get_presigned_url(
                        "faces", StorageKeys.face_crop(match.matched_face_id), expires_in=3600
                    )
                except Exception:
                    pass
            results.append(FaceSearchResult(
                face_id=match.matched_face_id or "",
                person_id=match.matched_person_id,
                person_name=None,
                similarity_score=round(match.similarity_score, 4),
                best_frame_url=face_url,
            ))
    return results


@router.get("/faces", response_model=list[FaceResponse])
async def list_faces(
    camera_id: Optional[str] = None,
    is_known: Optional[bool] = None,
    cluster_id: Optional[str] = None,
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """List detected faces. Filter by is_known=false to see unknowns awaiting ID."""
    faces = await FaceRepository.list(
        db, camera_id=camera_id, is_known=is_known,
        cluster_id=cluster_id, limit=limit, offset=offset,
    )
    storage = get_storage()
    result = []
    for f in faces:
        url = None
        if f.best_frame_path:
            try:
                url = await storage.get_presigned_url("faces", f.best_frame_path, expires_in=3600)
            except Exception:
                pass
        result.append(_face_resp(f, best_frame_url=url))
    return result


@router.get("/faces/{face_id}", response_model=FaceResponse)
async def get_face(
    face_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    face = await FaceRepository.get(db, face_id)
    if not face:
        raise HTTPException(404, f"Face '{face_id}' not found")
    storage = get_storage()
    url = None
    if face.best_frame_path:
        try:
            url = await storage.get_presigned_url("faces", face.best_frame_path, expires_in=3600)
        except Exception:
            pass
    return _face_resp(face, best_frame_url=url)


@router.patch("/faces/{face_id}/assign")
async def assign_face_to_person(
    face_id: str, person_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """Manually assign an unknown face to a known person and re-enroll in vector store."""
    face = await FaceRepository.get(db, face_id)
    if not face:
        raise HTTPException(404, f"Face '{face_id}' not found")
    person = await PersonRepository.get(db, person_id)
    if not person:
        raise HTTPException(404, f"Person '{person_id}' not found")

    # Update vector store metadata
    from core.abstractions.vector_store import get_vector_store
    store = get_vector_store()
    if face.embedding:
        await store.upsert(
            collection="faces", id=face_id,
            vector=list(face.embedding),
            metadata={"person_id": person_id, "face_id": face_id, "person_name": person.name},
        )

    # Update DB
    updated = await FaceRepository.assign_person(db, [face_id], person_id)
    return {"face_id": face_id, "person_id": person_id,
            "faces_updated": updated, "status": "assigned"}


@router.delete("/faces/{face_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face(
    face_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    from core.abstractions.vector_store import get_vector_store
    await get_vector_store().delete("faces", face_id)
    if not await FaceRepository.delete(db, face_id):
        raise HTTPException(404, f"Face '{face_id}' not found")


# ════════════════════════════════════════════════════
# DBSCAN CLUSTERING
# ════════════════════════════════════════════════════

@router.post("/faces/cluster/run")
async def run_clustering(_user: dict = Depends(get_current_user)):
    """Trigger DBSCAN face clustering manually. Also runs every 30 min via Celery beat."""
    from workers.celery_app import celery_app
    task = celery_app.send_task("workers.face_tasks.run_face_clustering")
    return {"task_id": task.id, "message": "Clustering task queued. Check GET /faces/clusters after ~1 min."}


@router.get("/faces/clusters", response_model=list[FaceClusterResponse])
async def list_clusters(
    min_size: int = 2,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """
    List DBSCAN clusters of visually similar unknown faces.
    Each cluster likely represents the same unidentified person.
    Assign clusters to persons via POST /faces/clusters/{cluster_id}/assign.
    """
    storage = get_storage()
    clusters = await FaceRepository.get_clusters_summary(db, min_size=min_size)
    result = []
    for c in clusters:
        url = None
        if c.get("representative_path"):
            try:
                url = await storage.get_presigned_url("faces", c["representative_path"], expires_in=3600)
            except Exception:
                pass
        result.append(FaceClusterResponse(
            cluster_id=c["cluster_id"],
            face_count=c["face_count"],
            representative_face_url=url,
            first_seen=c["first_seen"],
            last_seen=c["last_seen"],
            camera_ids=[],
        ))
    return result


@router.post("/faces/clusters/{cluster_id}/assign")
async def assign_cluster_to_person(
    cluster_id: str, person_id: str,
    db: AsyncSession = Depends(get_db), _user: dict = Depends(get_current_user),
):
    """Assign all faces in a cluster to a known person and update vector store."""
    person = await PersonRepository.get(db, person_id)
    if not person:
        raise HTTPException(404, f"Person '{person_id}' not found")

    # Get all faces in cluster
    faces = await FaceRepository.list(db, cluster_id=cluster_id, limit=1000)
    if not faces:
        raise HTTPException(404, f"Cluster '{cluster_id}' not found or empty")

    face_ids = [f.id for f in faces]

    # Update vector store for each face
    from core.abstractions.vector_store import get_vector_store
    store = get_vector_store()
    for face in faces:
        if face.embedding:
            try:
                await store.upsert(
                    collection="faces", id=face.id,
                    vector=list(face.embedding),
                    metadata={"person_id": person_id, "face_id": face.id, "person_name": person.name},
                )
            except Exception:
                pass

    # Update DB
    updated = await FaceRepository.assign_person(db, face_ids, person_id)
    log.info("cluster.assigned", cluster_id=cluster_id, person_id=person_id, faces=updated)

    return {"cluster_id": cluster_id, "person_id": person_id,
            "faces_updated": updated, "status": "assigned"}


# ─── Response helpers ──────────────────────────────────────────

def _person_resp(p, face_count: int) -> dict:
    return {"id": p.id, "name": p.name, "employee_id": p.employee_id,
            "department": p.department, "face_count": face_count,
            "is_watchlist": p.is_watchlist, "created_at": p.created_at}


def _face_resp(f, best_frame_url: Optional[str] = None) -> dict:
    return {
        "id": f.id, "person_id": f.person_id, "person_name": None,
        "cluster_id": f.cluster_id, "is_known": f.is_known,
        "best_frame_url": best_frame_url, "capture_count": f.capture_count,
        "quality_score": f.best_quality_score, "age_estimate": f.age_estimate,
        "gender": f.gender, "first_seen_at": f.first_seen_at,
        "last_seen_at": f.last_seen_at, "camera_ids": f.camera_ids or [],
    }
