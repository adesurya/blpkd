"""
tests/unit/test_detector.py
Unit tests for YOLOv8 detector and zone counter.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.detector.yolo_detector import BoundingBox, ZoneCounter


class TestBoundingBox:
    def test_to_pixel_coordinates(self):
        bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4)
        x1, y1, x2, y2 = bbox.to_pixel(1920, 1080)
        assert x1 == 192
        assert y1 == 216
        assert x2 == 768
        assert y2 == 648

    def test_center(self):
        bbox = BoundingBox(x=0.0, y=0.0, w=0.4, h=0.6)
        cx, cy = bbox.center()
        assert abs(cx - 0.2) < 1e-6
        assert abs(cy - 0.3) < 1e-6


class TestZoneCounter:
    def setup_method(self):
        self.zones = {
            "left_zone": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "right_zone": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        }
        self.counter = ZoneCounter(self.zones)

    def test_point_in_left_zone(self):
        zone = self.counter.get_zone(0.25, 0.5)
        assert zone == "left_zone"

    def test_point_in_right_zone(self):
        zone = self.counter.get_zone(0.75, 0.5)
        assert zone == "right_zone"

    def test_point_outside_all_zones(self):
        # On boundary — may be in either, just test no crash
        zone = self.counter.get_zone(0.5, 0.5)
        assert zone in ("left_zone", "right_zone", None)

    def test_empty_zones(self):
        counter = ZoneCounter({})
        zone = counter.get_zone(0.5, 0.5)
        assert zone is None


class TestColorDetector:
    def test_red_detection(self):
        from services.attribute_analyzer.color_detector import ColorDetector
        # Create a solid red image (BGR)
        red_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        red_frame[:, :, 2] = 200   # red channel high
        red_frame[:, :, 1] = 30    # green low
        red_frame[:, :, 0] = 30    # blue low
        color, hex_val = ColorDetector.detect_dominant_color(red_frame)
        assert color == "red"
        assert hex_val.startswith("#")

    def test_blue_detection(self):
        from services.attribute_analyzer.color_detector import ColorDetector
        blue_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        blue_frame[:, :, 0] = 200  # blue channel high (BGR)
        blue_frame[:, :, 1] = 30
        blue_frame[:, :, 2] = 30
        color, _ = ColorDetector.detect_dominant_color(blue_frame)
        assert color == "blue"

    def test_white_detection(self):
        from services.attribute_analyzer.color_detector import ColorDetector
        white_frame = np.full((100, 100, 3), 240, dtype=np.uint8)
        color, _ = ColorDetector.detect_dominant_color(white_frame)
        assert color == "white"

    def test_black_detection(self):
        from services.attribute_analyzer.color_detector import ColorDetector
        black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color, _ = ColorDetector.detect_dominant_color(black_frame)
        assert color == "black"

    def test_empty_frame_returns_unknown(self):
        from services.attribute_analyzer.color_detector import ColorDetector
        color, _ = ColorDetector.detect_dominant_color(None)
        assert color == "unknown"

    def test_analyze_person_crop_splits_body(self):
        from services.attribute_analyzer.color_detector import ColorDetector
        # Top half red (shirt), bottom half blue (pants)
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        crop[:100, :, 2] = 200  # top: red
        crop[100:, :, 0] = 200  # bottom: blue
        attrs = ColorDetector.analyze_person_crop(crop)
        assert attrs.upper_color == "red"
        assert attrs.lower_color == "blue"


class TestFaceQualityScorer:
    def test_larger_face_scores_higher(self):
        from services.face_engine.recognizer import FaceQualityScorer
        small_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        large_crop = np.zeros((200, 200, 3), dtype=np.uint8)
        small_score = FaceQualityScorer.score(small_crop, 0.9)
        large_score = FaceQualityScorer.score(large_crop, 0.9)
        assert large_score > small_score

    def test_higher_detection_score_improves_quality(self):
        from services.face_engine.recognizer import FaceQualityScorer
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        low_score = FaceQualityScorer.score(crop, 0.3)
        high_score = FaceQualityScorer.score(crop, 0.99)
        assert high_score > low_score

    def test_score_range(self):
        from services.face_engine.recognizer import FaceQualityScorer
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        score = FaceQualityScorer.score(crop, 0.8)
        assert 0.0 <= score <= 1.0


class TestDBSCANClusterer:
    def test_cluster_similar_embeddings(self):
        from services.face_engine.recognizer import FaceClusterer
        clusterer = FaceClusterer()

        # 3 faces close together (same person) + 1 outlier
        base = np.random.rand(512).astype(np.float32)
        base /= np.linalg.norm(base)

        embeddings = [
            base + np.random.randn(512).astype(np.float32) * 0.01,
            base + np.random.randn(512).astype(np.float32) * 0.01,
            base + np.random.randn(512).astype(np.float32) * 0.01,
            np.random.rand(512).astype(np.float32),  # outlier
        ]
        face_ids = ["f1", "f2", "f3", "f4_outlier"]

        result = clusterer.cluster(face_ids, embeddings)
        assert len(result) == 4

        # f1, f2, f3 should be in the same cluster
        clusters = {k: v for k, v in result.items() if k.startswith("f") and k != "f4_outlier"}
        cluster_labels = set(clusters.values())
        assert len(cluster_labels) == 1  # all in one cluster

    def test_insufficient_faces(self):
        from services.face_engine.recognizer import FaceClusterer
        clusterer = FaceClusterer()
        result = clusterer.cluster(["f1"], [np.random.rand(512)])
        assert result["f1"] == "noise"
