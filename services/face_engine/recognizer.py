"""
services/face_engine/recognizer.py

InsightFace-based face detection and recognition.
Features:
  - Face detection (SCRFD model)
  - Face embedding extraction (ArcFace/buffalo_l)
  - Face recognition via vector similarity search
  - DBSCAN clustering for unknown face grouping
  - Face quality scoring (for best capture selection)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import structlog
from sklearn.cluster import DBSCAN

from core.abstractions.vector_store import SearchResult, get_vector_store
from core.config.settings import settings

log = structlog.get_logger(__name__)

FACE_COLLECTION = "faces"


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    embedding: np.ndarray           # 512-dim ArcFace embedding
    detection_score: float
    crop: np.ndarray                # cropped face image
    kps: Optional[np.ndarray] = None  # 5 facial keypoints

    # InsightFace attributes (if analysis enabled)
    age: Optional[int] = None
    gender: Optional[str] = None  # "M" | "F"
    quality_score: float = 0.0

    @property
    def face_size(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class RecognitionResult:
    face: DetectedFace
    matched_person_id: Optional[str]  # None = unknown
    matched_face_id: Optional[str]
    similarity_score: float
    is_known: bool

    @property
    def label(self) -> str:
        if self.matched_person_id:
            return f"person:{self.matched_person_id}"
        return "unknown"


class FaceQualityScorer:
    """
    Score face quality for selecting best capture.
    Higher score = better quality (larger, sharper, more frontal).
    """

    @staticmethod
    def score(face_crop: np.ndarray, detection_score: float) -> float:
        h, w = face_crop.shape[:2]
        size_score = min(1.0, (w * h) / (112 * 112))  # normalize to target size

        # Sharpness via Laplacian variance
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 500.0)

        quality = (size_score * 0.4) + (detection_score * 0.4) + (sharpness_score * 0.2)
        return float(quality)


class FaceEngine:
    """
    Main face recognition engine.

    Pipeline per frame:
    1. detect_faces() — find all faces in frame
    2. extract_embeddings() — get 512-dim vector per face
    3. recognize() — match against known faces in vector store
    4. store_unknown() — save unrecognized faces for clustering later

    Scale note:
    - MVP: direct InsightFace call
    - Enterprise: replace with Triton Inference Server
    """

    _instance: Optional["FaceEngine"] = None

    def __init__(self) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        log.info("face_engine.loading", model=settings.INSIGHTFACE_MODEL)

        ctx_id = 0 if settings.insightface_device_resolved == "cuda" else -1
        self._app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL,
            root=settings.MODELS_PATH,
            allowed_modules=["detection", "recognition"],  # skip age/gender for speed
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._quality_scorer = FaceQualityScorer()
        self._vector_store = get_vector_store()

        log.info("face_engine.ready", model=settings.INSIGHTFACE_MODEL, ctx_id=ctx_id)

    @classmethod
    def get_instance(cls) -> "FaceEngine":
        """Singleton — model loading is expensive."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect_faces(
        self,
        frame: np.ndarray,
        min_size: int = None,
    ) -> list[DetectedFace]:
        """
        Detect all faces in a frame.

        Args:
            frame: BGR numpy array
            min_size: minimum face size in pixels (filter tiny/blurry faces)

        Returns:
            List of DetectedFace objects, sorted by detection_score DESC
        """
        min_size = min_size or settings.FACE_DETECTION_MIN_SIZE
        insightface_faces = self._app.get(frame)

        faces: list[DetectedFace] = []
        for f in insightface_faces:
            bbox = tuple(f.bbox.astype(int))
            x1, y1, x2, y2 = bbox
            face_w = x2 - x1
            face_h = y2 - y1

            # Filter too-small faces (low resolution CCTV)
            if face_w < min_size or face_h < min_size:
                continue

            # Crop face
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            quality = self._quality_scorer.score(crop, float(f.det_score))

            face = DetectedFace(
                bbox=(x1, y1, x2, y2),
                embedding=f.embedding,  # 512-dim numpy array
                detection_score=float(f.det_score),
                crop=crop,
                kps=f.kps,
                quality_score=quality,
            )
            faces.append(face)

        # Sort by quality (best first)
        faces.sort(key=lambda f: f.quality_score, reverse=True)
        return faces

    async def recognize(
        self,
        face: DetectedFace,
        threshold: float = None,
    ) -> RecognitionResult:
        """
        Recognize a face by searching the vector store.

        Args:
            face: DetectedFace with embedding
            threshold: minimum similarity score (default from settings)

        Returns:
            RecognitionResult with match info or unknown
        """
        threshold = threshold or settings.FACE_RECOGNITION_THRESHOLD
        embedding_list = face.embedding.tolist()

        results: list[SearchResult] = await self._vector_store.search(
            collection=FACE_COLLECTION,
            vector=embedding_list,
            top_k=1,
            threshold=threshold,
        )

        if results:
            best = results[0]
            return RecognitionResult(
                face=face,
                matched_person_id=best.metadata.get("person_id"),
                matched_face_id=best.id,
                similarity_score=best.score,
                is_known=True,
            )

        return RecognitionResult(
            face=face,
            matched_person_id=None,
            matched_face_id=None,
            similarity_score=0.0,
            is_known=False,
        )

    async def register_face(
        self,
        face: DetectedFace,
        face_id: str,
        person_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a face to the recognition database.

        Call this when:
        - Adding known persons (enrollment)
        - After DBSCAN clustering confirms a new cluster identity
        """
        await self._vector_store.upsert(
            collection=FACE_COLLECTION,
            id=face_id,
            vector=face.embedding.tolist(),
            metadata={
                "person_id": person_id,
                "face_id": face_id,
                **(metadata or {}),
            },
        )
        log.info("face_engine.face_registered", face_id=face_id, person_id=person_id)

    async def delete_face(self, face_id: str) -> None:
        await self._vector_store.delete(FACE_COLLECTION, face_id)

    def process_frame(
        self,
        frame: np.ndarray,
        person_crops: Optional[list[tuple[np.ndarray, int]]] = None,
    ) -> list[DetectedFace]:
        """
        Convenience: detect faces in full frame OR in person crops.

        When person_crops provided (from YOLOv8 detection), searches
        only within person bounding boxes — much faster & fewer false positives.
        """
        if person_crops:
            all_faces = []
            for crop, track_id in person_crops:
                if crop is None or crop.size == 0:
                    continue
                faces = self.detect_faces(crop)
                all_faces.extend(faces)
            return all_faces
        return self.detect_faces(frame)


# ─────────────────────────────────────────────────────────────
# DBSCAN Face Clusterer
# ─────────────────────────────────────────────────────────────
class FaceClusterer:
    """
    Groups unknown faces into clusters using DBSCAN.

    This runs as a background periodic job (e.g., every 30 minutes).
    Clusters unknown faces → suggests identities → human review can confirm.

    Training-free: clusters based on visual similarity of face embeddings.
    """

    def __init__(self) -> None:
        self.eps = settings.DBSCAN_EPS
        self.min_samples = settings.DBSCAN_MIN_SAMPLES

    def cluster(
        self,
        face_ids: list[str],
        embeddings: list[np.ndarray],
    ) -> dict[str, str]:
        """
        Cluster faces using DBSCAN on face embeddings.

        Args:
            face_ids: list of face IDs
            embeddings: corresponding 512-dim embeddings

        Returns:
            dict mapping face_id → cluster_label
            Noise points get label "noise"
        """
        if len(embeddings) < self.min_samples:
            log.warning("clusterer.not_enough_faces", count=len(embeddings))
            return {fid: "noise" for fid in face_ids}

        X = np.array(embeddings)
        # Normalize embeddings to unit sphere (cosine similarity becomes euclidean distance)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-10)

        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(X_norm)

        result = {}
        for face_id, label in zip(face_ids, labels):
            cluster_label = f"cluster_{label}" if label >= 0 else "noise"
            result[face_id] = cluster_label

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        log.info(
            "clusterer.complete",
            total=len(face_ids),
            clusters=n_clusters,
            noise=n_noise,
        )
        return result

    def suggest_merge_threshold(
        self,
        embeddings: list[np.ndarray],
        candidate_threshold: float = 0.7,
    ) -> list[tuple[int, int, float]]:
        """
        Find pairs of faces that should be merged (same person, different cluster).
        Returns list of (idx_a, idx_b, similarity).
        """
        X = np.array(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-10)

        # Cosine similarity matrix
        sim_matrix = X_norm @ X_norm.T
        pairs = []
        n = len(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= candidate_threshold:
                    pairs.append((i, j, sim))
        return sorted(pairs, key=lambda x: x[2], reverse=True)