"""
services/detector/yolo_detector.py

YOLOv8-based people detection with ByteTrack multi-object tracking.
Handles: detection, tracking, people counting, zone-based counting.

Designed to be model-agnostic — swap YOLOv8n → YOLOv8l by changing settings.
Custom trained models (clothing classifier, activity) slot in here too.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import structlog
from ultralytics import YOLO

from core.config.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class BoundingBox:
    """Normalized bounding box (0.0 to 1.0 relative to frame size)."""
    x: float   # top-left x
    y: float   # top-left y
    w: float   # width
    h: float   # height

    def to_pixel(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates."""
        return (
            int(self.x * frame_w),
            int(self.y * frame_h),
            int((self.x + self.w) * frame_w),
            int((self.y + self.h) * frame_h),
        )

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class PersonDetection:
    track_id: int
    bbox: BoundingBox
    confidence: float
    class_name: str = "person"
    frame_number: int = 0
    zone_id: Optional[str] = None  # which counting zone
    crop: Optional[np.ndarray] = None  # cropped person image for attribute analysis


@dataclass
class FrameDetectionResult:
    frame_number: int
    timestamp: float
    detections: list[PersonDetection]
    frame_width: int
    frame_height: int
    inference_time_ms: float

    @property
    def person_count(self) -> int:
        return len(self.detections)

    @property
    def count_by_zone(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in self.detections:
            zone = det.zone_id or "default"
            counts[zone] = counts.get(zone, 0) + 1
        return counts


class ZoneCounter:
    """
    Polygon-based zone counter.
    Counts people inside defined polygon zones.

    zones format: {"zone_name": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
    Coordinates are normalized (0.0-1.0).
    """

    def __init__(self, zones: dict[str, list[list[float]]]) -> None:
        self.zones = zones

    def get_zone(self, cx: float, cy: float) -> Optional[str]:
        """Find which zone a point (cx, cy) belongs to."""
        for zone_name, polygon in self.zones.items():
            pts = np.array(polygon, dtype=np.float32)
            # cv2.pointPolygonTest: +dist if inside, -dist if outside
            result = cv2.pointPolygonTest(pts, (cx, cy), measureDist=False)
            if result >= 0:
                return zone_name
        return None


class YOLODetector:
    """
    Main detection engine using YOLOv8 + ByteTrack.

    Features:
    - Real-time person detection
    - Multi-object tracking (persistent track_id per person)
    - Zone-based people counting
    - Frame crop extraction (for downstream face/attribute analysis)
    - Pluggable custom models (for clothing, activity)

    Scale note:
    - MVP: single GPU inference, direct call
    - Enterprise: replace with Triton Inference Server client
    """

    _instance: Optional["YOLODetector"] = None

    def __init__(self) -> None:
        self.model_path = settings.YOLO_MODEL
        self.confidence = settings.YOLO_CONFIDENCE
        self.iou_threshold = settings.YOLO_IOU_THRESHOLD
        self.device = settings.YOLO_DEVICE

        log.info("detector.loading_model", model=self.model_path, device=self.device)
        self._model = YOLO(self.model_path)
        self._model.to(self.device)
        log.info("detector.model_ready", model=self.model_path)

        # Optional: load custom attribute model
        self._attribute_model: Optional[YOLO] = None
        self._custom_model_path = Path(settings.MODELS_PATH) / "custom_attribute.pt"
        if self._custom_model_path.exists():
            self._attribute_model = YOLO(str(self._custom_model_path))
            self._attribute_model.to(self.device)
            log.info("detector.custom_model_loaded", path=str(self._custom_model_path))

    @classmethod
    def get_instance(cls) -> "YOLODetector":
        """Singleton — models are expensive to load."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        zones: Optional[dict[str, list[list[float]]]] = None,
        extract_crops: bool = True,
    ) -> FrameDetectionResult:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR numpy array (from cv2.VideoCapture or PyAV)
            frame_number: sequential frame index
            timestamp: video timestamp in seconds
            zones: polygon zones for zone-based counting
            extract_crops: whether to extract cropped person images

        Returns:
            FrameDetectionResult with all detections
        """
        h, w = frame.shape[:2]
        t_start = time.perf_counter()

        # Run YOLOv8 with ByteTrack tracker
        # persist=True keeps tracks across frames
        results = self._model.track(
            frame,
            persist=True,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            classes=[0],         # class 0 = "person" in COCO
            verbose=False,
            tracker="bytetrack.yaml",   # ByteTrack (fast) or botsort.yaml (more accurate)
        )

        inference_ms = (time.perf_counter() - t_start) * 1000

        zone_counter = ZoneCounter(zones) if zones else None
        detections: list[PersonDetection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                # Normalized xywh
                xywhn = box.xywhn[0].cpu().numpy()
                cx_norm = float(xywhn[0])
                cy_norm = float(xywhn[1])
                w_norm = float(xywhn[2])
                h_norm = float(xywhn[3])

                bbox = BoundingBox(
                    x=cx_norm - w_norm / 2,
                    y=cy_norm - h_norm / 2,
                    w=w_norm,
                    h=h_norm,
                )

                track_id = int(box.id[0]) if box.id is not None else i
                confidence = float(box.conf[0])

                # Determine zone
                zone_id = None
                if zone_counter:
                    zone_id = zone_counter.get_zone(cx_norm, cy_norm)

                # Extract crop for attribute/face analysis
                crop = None
                if extract_crops:
                    x1, y1, x2, y2 = bbox.to_pixel(w, h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()

                detections.append(PersonDetection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    frame_number=frame_number,
                    zone_id=zone_id,
                    crop=crop,
                ))

        result = FrameDetectionResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            frame_width=w,
            frame_height=h,
            inference_time_ms=inference_ms,
        )

        log.debug(
            "detector.frame_processed",
            frame=frame_number,
            count=result.person_count,
            inference_ms=round(inference_ms, 1),
        )

        return result

    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_numbers: list[int],
        timestamps: list[float],
    ) -> list[FrameDetectionResult]:
        """
        Batch detection for higher GPU utilization.
        Use this for recorded video processing (not realtime stream).
        """
        results = []
        for frame, fn, ts in zip(frames, frame_numbers, timestamps):
            results.append(self.detect(frame, fn, ts))
        return results

    def run_custom_classifier(
        self, crop: np.ndarray
    ) -> Optional[dict[str, float]]:
        """
        Run custom trained attribute classifier on person crop.
        Returns class probabilities.

        Used for: clothing type, activity classification.
        Train your custom model with scripts/training/train_clothing_classifier.py
        """
        if self._attribute_model is None:
            return None

        results = self._model(crop, verbose=False)
        if results and results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()
            names = results[0].names
            return {names[i]: float(probs[i]) for i in range(len(probs))}
        return None

    def warmup(self) -> None:
        """
        Run a dummy inference to warm up GPU.
        Call once before starting stream processing.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, extract_crops=False)
        log.info("detector.warmup_complete")
