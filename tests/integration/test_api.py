"""
tests/integration/test_api.py
Integration tests for the FastAPI endpoints.
Run with: pytest tests/integration/ -v
Requires running PostgreSQL + Redis (use docker compose test profile).
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from services.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "changeme123"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_unauthenticated_api_returns_401(self, client):
        resp = client.get("/api/v1/cameras")
        assert resp.status_code == 401


class TestAuth:
    def test_login_success(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "changeme123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    def test_get_current_user(self, client, auth_headers):
        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"


class TestCameras:
    def test_create_camera_rtsp(self, client, auth_headers):
        resp = client.post(
            "/api/v1/cameras",
            headers=auth_headers,
            json={
                "name": "Test Camera",
                "source_type": "rtsp",
                "source_url": "rtsp://192.168.1.1:554/stream",
                "frame_sample_rate": 5,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Camera"
        assert data["source_type"] == "rtsp"
        assert "id" in data

    def test_create_camera_m3u8(self, client, auth_headers):
        resp = client.post(
            "/api/v1/cameras",
            headers=auth_headers,
            json={
                "name": "HLS Stream Camera",
                "source_type": "m3u8",
                "source_url": "https://example.com/live/cam01.m3u8",
                "frame_sample_rate": 10,
            },
        )
        assert resp.status_code == 201

    def test_create_camera_with_zones(self, client, auth_headers):
        resp = client.post(
            "/api/v1/cameras",
            headers=auth_headers,
            json={
                "name": "Zoned Camera",
                "source_type": "rtsp",
                "source_url": "rtsp://192.168.1.1:554/stream",
                "zones": {
                    "entrance": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                    "exit": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "entrance" in data["zones"]

    def test_list_cameras(self, client, auth_headers):
        resp = client.get("/api/v1/cameras", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_camera_not_found(self, client, auth_headers):
        resp = client.get("/api/v1/cameras/nonexistent_id", headers=auth_headers)
        assert resp.status_code == 404


class TestPersonEnrollment:
    def test_create_person(self, client, auth_headers):
        resp = client.post(
            "/api/v1/persons",
            headers=auth_headers,
            json={
                "name": "Test Person",
                "employee_id": "EMP001",
                "department": "Engineering",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Person"
        assert data["face_count"] == 0

    def test_create_watchlist_person(self, client, auth_headers):
        resp = client.post(
            "/api/v1/persons",
            headers=auth_headers,
            json={"name": "VIP Guest", "is_watchlist": True},
        )
        assert resp.status_code == 201
        assert resp.json()["is_watchlist"] is True

    def test_list_persons(self, client, auth_headers):
        resp = client.get("/api/v1/persons", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestVideoUpload:
    def test_upload_unsupported_format(self, client, auth_headers):
        """Test that non-video files are rejected."""
        resp = client.post(
            "/api/v1/videos/upload",
            headers=auth_headers,
            files={"file": ("document.pdf", b"fake pdf content", "application/pdf")},
        )
        assert resp.status_code == 415

    def test_upload_video_returns_recording_id(self, client, auth_headers):
        """Test that valid video upload returns job IDs."""
        # Minimal valid MP4 header bytes (just for format check)
        fake_mp4 = b"\x00\x00\x00\x1cftyp" + b"\x00" * 100
        resp = client.post(
            "/api/v1/videos/upload",
            headers=auth_headers,
            files={"file": ("test_video.mp4", fake_mp4, "video/mp4")},
            data={"extract_faces": "true", "compress": "true"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "recording_id" in data
        assert "task_id" in data
        assert data["status"] == "queued"


class TestAnalytics:
    def test_live_counts_empty(self, client, auth_headers):
        resp = client.get("/api/v1/analytics/count/live", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_timeseries_requires_camera_id(self, client, auth_headers):
        resp = client.get(
            "/api/v1/analytics/count/timeseries",
            headers=auth_headers,
        )
        assert resp.status_code == 422  # missing required param

    def test_timeseries_valid_request(self, client, auth_headers):
        resp = client.get(
            "/api/v1/analytics/count/timeseries",
            headers=auth_headers,
            params={"camera_id": "cam_test", "interval_minutes": 15},
        )
        assert resp.status_code == 200


class TestWebhooks:
    def test_create_webhook(self, client, auth_headers):
        resp = client.post(
            "/api/v1/webhooks",
            headers=auth_headers,
            json={
                "name": "Test Webhook",
                "url": "https://example.com/hook",
                "events": ["face.recognized", "person.detected"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Webhook"
        assert "face.recognized" in data["events"]

    def test_create_webhook_invalid_event(self, client, auth_headers):
        resp = client.post(
            "/api/v1/webhooks",
            headers=auth_headers,
            json={
                "name": "Bad Webhook",
                "url": "https://example.com/hook",
                "events": ["invalid.event.type"],
            },
        )
        assert resp.status_code == 400

    def test_list_webhooks(self, client, auth_headers):
        resp = client.get("/api/v1/webhooks", headers=auth_headers)
        assert resp.status_code == 200
