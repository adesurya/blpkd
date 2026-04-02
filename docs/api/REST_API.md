# Vision Platform — REST API Documentation

Base URL: `http://your-server/api/v1`
Auth: Bearer JWT token (get via `POST /auth/token`)

---

## Authentication

### POST /auth/token
Get access token.

**Request:**
```
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

username=admin&password=changeme123
```

**Response 200:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

All subsequent requests require:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Cameras & Streams

### POST /cameras — Register a Camera

**Request:**
```json
{
  "name": "Lobby Entrance A",
  "description": "Main lobby entrance, facing north",
  "location": "Building A - Ground Floor",
  "source_type": "rtsp",
  "source_url": "rtsp://admin:password@192.168.1.100:554/stream1",
  "fps_target": 5,
  "frame_sample_rate": 5,
  "zones": {
    "entrance": [[0.0,0.0],[0.5,0.0],[0.5,1.0],[0.0,1.0]],
    "exit":     [[0.5,0.0],[1.0,0.0],[1.0,1.0],[0.5,1.0]]
  }
}
```

**Response 201:**
```json
{
  "id": "cam_7f3a9b2c",
  "name": "Lobby Entrance A",
  "description": "Main lobby entrance, facing north",
  "location": "Building A - Ground Floor",
  "source_type": "rtsp",
  "source_url": "rtsp://admin:password@192.168.1.100:554/stream1",
  "fps_target": 5,
  "is_active": true,
  "is_recording": false,
  "last_seen_at": null,
  "zones": {
    "entrance": [[0.0,0.0],[0.5,0.0],[0.5,1.0],[0.0,1.0]],
    "exit":     [[0.5,0.0],[1.0,0.0],[1.0,1.0],[0.5,1.0]]
  },
  "created_at": "2026-03-31T08:00:00Z"
}
```

---

### POST /cameras — Register m3u8 HLS Stream

**Request:**
```json
{
  "name": "Mall Entrance Live",
  "source_type": "m3u8",
  "source_url": "https://cdn.example.com/live/camera01.m3u8",
  "frame_sample_rate": 10
}
```

---

### POST /streams/{camera_id}/start — Start Stream Analysis

**Request:**
```json
{
  "camera_id": "cam_7f3a9b2c",
  "extract_faces": true,
  "analyze_attributes": true,
  "filter_criteria": null
}
```

**With attribute filter (count only blue-shirt people):**
```json
{
  "camera_id": "cam_7f3a9b2c",
  "extract_faces": true,
  "analyze_attributes": true,
  "filter_criteria": {
    "upper_color": "blue"
  }
}
```

**Response 200:**
```json
{
  "camera_id": "cam_7f3a9b2c",
  "is_running": true,
  "frames_processed": 0,
  "current_count": 0,
  "started_at": "2026-03-31T08:00:00Z"
}
```

---

### GET /streams/{camera_id}/status — Live Status

**Response 200:**
```json
{
  "camera_id": "cam_7f3a9b2c",
  "is_running": true,
  "frames_processed": 1240,
  "current_count": 7,
  "started_at": "2026-03-31T08:00:00Z"
}
```

---

## Video Processing

### POST /videos/upload — Upload Video for Analysis

```
POST /api/v1/videos/upload
Content-Type: multipart/form-data

file=@recording_2026.mp4
camera_id=cam_7f3a9b2c
extract_faces=true
analyze_attributes=true
compress=true
sample_rate=5
zones={"entrance":[[0,0],[0.5,0],[0.5,1],[0,1]]}
```

**Response 202:**
```json
{
  "recording_id": "rec_d4e5f6a7",
  "task_id": "celery-task-abc-123",
  "status": "queued",
  "message": "Video uploaded and queued for processing. Poll /videos/rec_d4e5f6a7/status for updates."
}
```

---

### GET /videos/{recording_id}/status — Poll Task Status

**Response (running):**
```json
{
  "task_id": "celery-task-abc-123",
  "status": "running",
  "progress": 0.45,
  "result": null,
  "error": null
}
```

**Response (completed):**
```json
{
  "task_id": "celery-task-abc-123",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "recording_id": "rec_d4e5f6a7",
    "total_frames_processed": 480,
    "total_persons_detected": 1523,
    "total_faces_detected": 234,
    "unique_persons_count": 67,
    "duration_seconds": 12.4,
    "compression_ratio": 4.8,
    "summary": [
      {
        "frame_number": 5,
        "timestamp": 0.2,
        "count": 3,
        "count_by_zone": {"entrance": 2, "exit": 1},
        "detections": [
          {
            "track_id": 1,
            "confidence": 0.91,
            "bbox": {"x": 0.12, "y": 0.08, "w": 0.18, "h": 0.64},
            "zone_id": "entrance",
            "upper_color": "blue",
            "lower_color": "black",
            "upper_color_hex": "#1A3B6E",
            "activity": "walking",
            "matches_filter": true
          }
        ]
      }
    ]
  },
  "error": null
}
```

---

### GET /videos/{recording_id}/download — Get Download URL

**Response 200:**
```json
{
  "recording_id": "rec_d4e5f6a7",
  "version": "compressed",
  "url": "https://storage.example.com/compressed/rec_d4e5f6a7.mp4?X-Amz-Expires=3600&...",
  "expires_in_seconds": 3600
}
```

---

## Face Recognition

### POST /persons — Create a Known Person

**Request:**
```json
{
  "name": "John Doe",
  "employee_id": "EMP-001",
  "department": "Engineering",
  "is_watchlist": false
}
```

**Response 201:**
```json
{
  "id": "person_a1b2c3d4",
  "name": "John Doe",
  "employee_id": "EMP-001",
  "department": "Engineering",
  "face_count": 0,
  "is_watchlist": false,
  "created_at": "2026-03-31T08:00:00Z"
}
```

---

### POST /persons/{person_id}/enroll — Enroll Face Photos

```
POST /api/v1/persons/person_a1b2c3d4/enroll
Content-Type: multipart/form-data

photos[]=@john_front.jpg
photos[]=@john_left.jpg
photos[]=@john_right.jpg
```

**Response 201:**
```json
{
  "person_id": "person_a1b2c3d4",
  "enrolled_count": 3,
  "errors": [],
  "message": "Successfully enrolled 3 face(s)."
}
```

**Response with errors (some photos invalid):**
```json
{
  "person_id": "person_a1b2c3d4",
  "enrolled_count": 2,
  "errors": [
    "photo_blurry.jpg: no face detected",
    "group_photo.jpg: multiple faces found, use single-face photos"
  ],
  "message": "Successfully enrolled 2 face(s)."
}
```

---

### POST /faces/search — Search Face by Photo

```
POST /api/v1/faces/search
Content-Type: multipart/form-data

photo=@unknown_person.jpg
threshold=0.45
top_k=5
```

**Response 200 (match found):**
```json
[
  {
    "face_id": "face_x9y8z7",
    "person_id": "person_a1b2c3d4",
    "person_name": "John Doe",
    "similarity_score": 0.87,
    "best_frame_url": "https://storage.example.com/faces/face_x9y8z7.jpg?..."
  }
]
```

**Response 200 (no match):**
```json
[]
```

---

### GET /faces — List Detected Faces

**Request:**
```
GET /api/v1/faces?is_known=false&limit=20
```

**Response 200:**
```json
[
  {
    "id": "face_u1v2w3x4",
    "person_id": null,
    "person_name": null,
    "cluster_id": "cluster_5",
    "is_known": false,
    "best_frame_url": "https://storage.example.com/faces/face_u1v2w3x4.jpg?...",
    "capture_count": 12,
    "quality_score": 0.78,
    "age_estimate": 32,
    "gender": "M",
    "first_seen_at": "2026-03-31T07:15:00Z",
    "last_seen_at": "2026-03-31T09:42:00Z",
    "camera_ids": ["cam_7f3a9b2c", "cam_2k3l4m5n"]
  }
]
```

---

### GET /faces/clusters — View DBSCAN Clusters

**Response 200:**
```json
[
  {
    "cluster_id": "cluster_5",
    "face_count": 12,
    "representative_face_url": "https://storage.example.com/faces/face_u1v2w3x4.jpg?...",
    "first_seen": "2026-03-31T07:15:00Z",
    "last_seen": "2026-03-31T09:42:00Z",
    "camera_ids": ["cam_7f3a9b2c"]
  },
  {
    "cluster_id": "cluster_8",
    "face_count": 7,
    "representative_face_url": "https://storage.example.com/faces/face_p5q6r7.jpg?...",
    "first_seen": "2026-03-31T08:00:00Z",
    "last_seen": "2026-03-31T10:00:00Z",
    "camera_ids": ["cam_2k3l4m5n"]
  }
]
```

---

### POST /faces/clusters/{cluster_id}/assign — Assign Cluster to Person

**Request:**
```json
{
  "person_id": "person_a1b2c3d4"
}
```

**Response 200:**
```json
{
  "cluster_id": "cluster_5",
  "person_id": "person_a1b2c3d4",
  "faces_updated": 12,
  "status": "assigned"
}
```

---

## Analytics & Counting

### GET /analytics/count/live — Live People Counts

**Response 200:**
```json
[
  {
    "camera_id": "cam_7f3a9b2c",
    "camera_name": "Lobby Entrance A",
    "zone_id": "entrance",
    "timestamp": "2026-03-31T10:05:00Z",
    "count": 7,
    "count_entering": 3,
    "count_exiting": 2,
    "count_by_upper_color": {
      "blue": 3,
      "white": 2,
      "black": 1,
      "red": 1
    }
  },
  {
    "camera_id": "cam_7f3a9b2c",
    "camera_name": "Lobby Entrance A",
    "zone_id": "exit",
    "timestamp": "2026-03-31T10:05:00Z",
    "count": 4,
    "count_entering": 1,
    "count_exiting": 3,
    "count_by_upper_color": {"white": 2, "gray": 2}
  }
]
```

---

### GET /analytics/count/timeseries — Count Time Series

**Request:**
```
GET /api/v1/analytics/count/timeseries
  ?camera_id=cam_7f3a9b2c
  &zone_id=entrance
  &start=2026-03-31T08:00:00Z
  &end=2026-03-31T12:00:00Z
  &interval_minutes=15
```

**Response 200:**
```json
{
  "camera_id": "cam_7f3a9b2c",
  "zone_id": "entrance",
  "start_time": "2026-03-31T08:00:00Z",
  "end_time": "2026-03-31T12:00:00Z",
  "interval_minutes": 15,
  "data": [
    {"timestamp": "2026-03-31T08:00:00Z", "count": 5, "entering": 3, "exiting": 2},
    {"timestamp": "2026-03-31T08:15:00Z", "count": 8, "entering": 6, "exiting": 3},
    {"timestamp": "2026-03-31T08:30:00Z", "count": 12, "entering": 9, "exiting": 5},
    {"timestamp": "2026-03-31T08:45:00Z", "count": 7, "entering": 2, "exiting": 7}
  ]
}
```

---

### GET /analytics/summary/{camera_id} — Analytics Summary

**Response 200:**
```json
{
  "camera_id": "cam_7f3a9b2c",
  "period_start": "2026-03-31T08:00:00Z",
  "period_end": "2026-03-31T18:00:00Z",
  "total_detections": 8420,
  "unique_track_ids": 312,
  "peak_count": 24,
  "peak_time": "2026-03-31T12:15:00Z",
  "average_count": 8.4,
  "color_distribution": {
    "blue": 2145,
    "white": 1890,
    "black": 1654,
    "gray": 987,
    "red": 445
  },
  "zone_breakdown": {
    "entrance": 5234,
    "exit": 3186
  }
}
```

---

## Webhooks

### POST /webhooks — Register Webhook

**Request:**
```json
{
  "name": "Security Alert System",
  "url": "https://your-app.com/hooks/vision-events",
  "secret": "my-webhook-secret-key-256bit",
  "events": [
    "face.recognized",
    "face.unknown",
    "count.threshold"
  ],
  "camera_ids": ["cam_7f3a9b2c"]
}
```

**Response 201:**
```json
{
  "id": "wh_q1r2s3t4",
  "name": "Security Alert System",
  "url": "https://your-app.com/hooks/vision-events",
  "events": ["face.recognized", "face.unknown", "count.threshold"],
  "camera_ids": ["cam_7f3a9b2c"],
  "is_active": true,
  "created_at": "2026-03-31T08:00:00Z"
}
```

---

### Webhook Payloads (Received by Your Endpoint)

**face.recognized event:**
```json
{
  "event": "face.recognized",
  "timestamp": "2026-03-31T10:05:23Z",
  "camera_id": "cam_7f3a9b2c",
  "person_id": "person_a1b2c3d4",
  "person_name": "John Doe",
  "similarity_score": 0.87,
  "face_crop_url": "https://storage.example.com/faces/tmp_abc.jpg?expires=1200"
}
```

**face.unknown event:**
```json
{
  "event": "face.unknown",
  "timestamp": "2026-03-31T10:07:45Z",
  "camera_id": "cam_7f3a9b2c",
  "face_id": "face_u1v2w3x4",
  "face_crop_url": "https://storage.example.com/faces/face_u1v2w3x4.jpg?expires=3600"
}
```

**person.detected event:**
```json
{
  "event": "person.detected",
  "timestamp": "2026-03-31T10:05:00Z",
  "camera_id": "cam_7f3a9b2c",
  "camera_name": "Lobby Entrance A",
  "frame_number": 1450,
  "detections": [
    {
      "track_id": 42,
      "confidence": 0.93,
      "bbox": {"x": 0.25, "y": 0.1, "w": 0.15, "h": 0.6},
      "zone_id": "entrance",
      "upper_color": "blue",
      "lower_color": "black"
    }
  ],
  "total_count": 7,
  "zone_counts": {"entrance": 5, "exit": 2}
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Human-readable error message"
}
```

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Bad request / validation error |
| 401 | Unauthorized — missing/invalid token |
| 404 | Resource not found |
| 415 | Unsupported media type (wrong file format) |
| 422 | Unprocessable entity (e.g., no face in photo) |
| 429 | Rate limit exceeded (100 req/min per IP) |
| 500 | Internal server error |
| 503 | Service unavailable (model not loaded) |
