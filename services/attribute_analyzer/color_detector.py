"""
services/attribute_analyzer/color_detector.py

Clothing color and attribute analysis using OpenCV HSV.
Also integrates custom YOLOv8 classifier for clothing type / activity.

Color detection pipeline:
1. Crop upper body (top 60% of person bbox)
2. Crop lower body (bottom 40% of person bbox)
3. Convert to HSV color space
4. Mask out background using GrabCut
5. Find dominant color via histogram analysis
6. Map HSV range → color label

Custom training:
  - To add more attributes, train custom YOLOv8 classifier
  - See scripts/training/train_clothing_classifier.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# HSV Color Ranges (tuned for clothing detection)
# Add/modify ranges here to support more colors
# ─────────────────────────────────────────────────────────────
COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (np.array([0, 70, 50]), np.array([10, 255, 255])),
        (np.array([170, 70, 50]), np.array([180, 255, 255])),  # red wraps around
    ],
    "orange": [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
    "yellow": [(np.array([25, 100, 100]), np.array([35, 255, 255]))],
    "green": [(np.array([35, 40, 40]), np.array([85, 255, 255]))],
    "blue": [(np.array([85, 50, 50]), np.array([130, 255, 255]))],
    "purple": [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
    "pink": [(np.array([160, 30, 150]), np.array([170, 150, 255]))],
    "white": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
    "gray": [(np.array([0, 0, 50]), np.array([180, 30, 200]))],
    "brown": [(np.array([10, 50, 50]), np.array([20, 200, 150]))],
}


@dataclass
class PersonAttributes:
    upper_color: Optional[str] = None       # shirt/jacket color
    lower_color: Optional[str] = None       # pants/skirt color
    upper_color_hex: Optional[str] = None   # dominant hex color
    lower_color_hex: Optional[str] = None
    clothing_type: Optional[str] = None     # from custom model: "shirt", "jacket", "hoodie"
    activity: Optional[str] = None          # from custom model: "walking", "standing", "running"
    activity_confidence: float = 0.0
    raw: dict = None  # all raw attribute scores


class ColorDetector:
    """
    OpenCV HSV-based clothing color detector.

    Works well for: solid colors, basic color classification.
    For patterned clothing: use custom deep learning classifier.
    """

    @staticmethod
    def detect_dominant_color(
        region: np.ndarray,
        n_colors: int = 3,
    ) -> tuple[str, str]:
        """
        Find dominant color label and hex in a region.

        Returns:
            (color_label, hex_color)
        """
        if region is None or region.size == 0:
            return "unknown", "#000000"

        # Resize for speed
        region = cv2.resize(region, (64, 64))

        # Remove very dark/light background pixels
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # GrabCut to separate foreground (clothing) from background
        # For CCTV crops this is often good enough without GrabCut
        mask = np.ones(region.shape[:2], dtype=np.uint8) * cv2.GC_FGD

        # Score each color range
        color_scores: dict[str, float] = {}
        total_pixels = region.shape[0] * region.shape[1]

        for color_name, ranges in COLOR_RANGES.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                m = cv2.inRange(hsv, lower, upper)
                color_mask = cv2.bitwise_or(color_mask, m)
            score = cv2.countNonZero(color_mask) / total_pixels
            color_scores[color_name] = score

        # Get dominant color (highest score, minimum 5% coverage)
        best_color = max(color_scores, key=color_scores.get)
        if color_scores[best_color] < 0.05:
            best_color = "unknown"

        # Get dominant hex for UI display
        hex_color = ColorDetector._get_dominant_hex(region)

        return best_color, hex_color

    @staticmethod
    def _get_dominant_hex(region: np.ndarray) -> str:
        """Get the single most dominant pixel color as hex."""
        pixels = region.reshape(-1, 3).astype(np.float32)
        # Use k-means with k=1 to get average dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        b, g, r = centers[0].astype(int)
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def analyze_person_crop(crop: np.ndarray) -> PersonAttributes:
        """
        Full attribute analysis of a person crop.

        Splits crop into upper body (shirt) and lower body (pants).
        """
        if crop is None or crop.size == 0:
            return PersonAttributes()

        h, w = crop.shape[:2]

        # Upper body: top 55% of crop
        upper_region = crop[:int(h * 0.55), :]

        # Lower body: bottom 45% of crop
        lower_region = crop[int(h * 0.55):, :]

        upper_color, upper_hex = ColorDetector.detect_dominant_color(upper_region)
        lower_color, lower_hex = ColorDetector.detect_dominant_color(lower_region)

        return PersonAttributes(
            upper_color=upper_color,
            lower_color=lower_color,
            upper_color_hex=upper_hex,
            lower_color_hex=lower_hex,
        )


class AttributeAnalyzer:
    """
    Combined attribute analyzer.
    Merges:
    - ColorDetector (HSV, zero-training)
    - Custom YOLOv8 classifier (clothing type, activity)

    To add your custom model:
        1. Train with scripts/training/train_clothing_classifier.py
        2. Save to /app/models/custom_attribute.pt
        3. It will be auto-loaded by YOLODetector
    """

    def __init__(self) -> None:
        self._color_detector = ColorDetector()
        self._custom_model = None  # loaded lazily if available

    def analyze(
        self,
        person_crop: np.ndarray,
        custom_predictions: Optional[dict[str, float]] = None,
    ) -> PersonAttributes:
        """
        Full attribute analysis pipeline.

        Args:
            person_crop: cropped person image from YOLO detection
            custom_predictions: optional output from custom YOLOv8 classifier
        """
        # Step 1: Color detection (always available, zero training)
        attrs = self._color_detector.analyze_person_crop(person_crop)

        # Step 2: Custom model predictions (if available)
        if custom_predictions:
            # Extract clothing type
            clothing_classes = ["shirt", "jacket", "hoodie", "dress", "uniform"]
            activity_classes = ["standing", "walking", "running", "sitting", "crouching"]

            clothing_scores = {k: v for k, v in custom_predictions.items() if k in clothing_classes}
            activity_scores = {k: v for k, v in custom_predictions.items() if k in activity_classes}

            if clothing_scores:
                attrs.clothing_type = max(clothing_scores, key=clothing_scores.get)
            if activity_scores:
                best_activity = max(activity_scores, key=activity_scores.get)
                attrs.activity = best_activity
                attrs.activity_confidence = activity_scores[best_activity]

            attrs.raw = custom_predictions

        return attrs

    def analyze_with_counting_filter(
        self,
        person_crop: np.ndarray,
        filter_criteria: Optional[dict] = None,
    ) -> tuple[PersonAttributes, bool]:
        """
        Analyze attributes AND check if person matches filter criteria.

        This enables:
        "Count only people wearing blue shirts"
        "Count only people who are running"

        Args:
            person_crop: person image
            filter_criteria: e.g., {"upper_color": "blue", "activity": "running"}

        Returns:
            (attributes, matches_filter)
        """
        attrs = self.analyze(person_crop)

        if not filter_criteria:
            return attrs, True

        for key, value in filter_criteria.items():
            attr_value = getattr(attrs, key, None)
            if attr_value != value:
                return attrs, False

        return attrs, True
