-- storage/migrations/init.sql
-- Runs once on first postgres container start

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for text search

-- ─────────────────────────────────────────────────
-- API Users
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_users (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    username        TEXT UNIQUE NOT NULL,
    email           TEXT,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',  -- admin | operator | viewer
    is_active       BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Default admin user (password: changeme123 — CHANGE IN PRODUCTION)
INSERT INTO api_users (username, email, hashed_password, role)
VALUES ('admin', 'admin@vision-platform.local',
        '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', -- "changeme123"
        'admin')
ON CONFLICT DO NOTHING;

-- ─────────────────────────────────────────────────
-- Cameras
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cameras (
    id                  TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name                TEXT NOT NULL,
    description         TEXT,
    location            TEXT,
    source_type         TEXT NOT NULL,  -- rtsp | m3u8 | file | webcam
    source_url          TEXT NOT NULL,
    fps_target          INT NOT NULL DEFAULT 5,
    frame_sample_rate   INT NOT NULL DEFAULT 5,
    resolution_width    INT,
    resolution_height   INT,
    is_active           BOOLEAN NOT NULL DEFAULT true,
    is_recording        BOOLEAN NOT NULL DEFAULT false,
    last_seen_at        TIMESTAMPTZ,
    zones               JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ
);

-- ─────────────────────────────────────────────────
-- Detection Sessions
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS detection_sessions (
    id                      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    camera_id               TEXT REFERENCES cameras(id) ON DELETE SET NULL,
    status                  TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed | stopped
    started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at                TIMESTAMPTZ,
    total_frames_processed  INT NOT NULL DEFAULT 0,
    total_detections        INT NOT NULL DEFAULT 0,
    total_faces_detected    INT NOT NULL DEFAULT 0,
    total_faces_recognized  INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_camera ON detection_sessions(camera_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON detection_sessions(status);

-- ─────────────────────────────────────────────────
-- Persons (known identities)
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persons (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name            TEXT NOT NULL,
    employee_id     TEXT,
    department      TEXT,
    metadata        JSONB,
    is_watchlist    BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_persons_watchlist ON persons(is_watchlist) WHERE is_watchlist = true;

-- ─────────────────────────────────────────────────
-- Faces
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS faces (
    id                  TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    person_id           TEXT REFERENCES persons(id) ON DELETE SET NULL,
    cluster_id          TEXT,
    is_known            BOOLEAN NOT NULL DEFAULT false,
    best_frame_path     TEXT,
    capture_count       INT NOT NULL DEFAULT 1,
    best_quality_score  FLOAT,
    embedding           vector(512),   -- InsightFace ArcFace embedding
    age_estimate        INT,
    gender              TEXT,          -- M | F
    detection_score     FLOAT,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    camera_ids          JSONB          -- [camera_id, ...]
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_known ON faces(is_known);
-- IVFFlat index for approximate nearest neighbor search
-- Adjust lists parameter based on data size: sqrt(total_rows)
CREATE INDEX IF NOT EXISTS idx_faces_embedding ON faces
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ─────────────────────────────────────────────────
-- Detections (one row = one person in one frame)
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS detections (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id      TEXT REFERENCES detection_sessions(id) ON DELETE CASCADE,
    camera_id       TEXT NOT NULL,
    frame_number    BIGINT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    frame_path      TEXT,
    -- Bounding box (normalized 0-1)
    bbox_x          FLOAT NOT NULL,
    bbox_y          FLOAT NOT NULL,
    bbox_w          FLOAT NOT NULL,
    bbox_h          FLOAT NOT NULL,
    -- Tracking
    track_id        INT,
    confidence      FLOAT NOT NULL,
    class_name      TEXT NOT NULL DEFAULT 'person',
    -- Attributes
    upper_color     TEXT,
    lower_color     TEXT,
    clothing_type   TEXT,
    activity        TEXT,
    attributes      JSONB,
    -- Zone
    zone_id         TEXT,
    -- Face link
    face_id         TEXT REFERENCES faces(id) ON DELETE SET NULL,
    face_confidence FLOAT
) PARTITION BY RANGE (timestamp);

-- Create initial partition (current month)
CREATE TABLE IF NOT EXISTS detections_default PARTITION OF detections DEFAULT;

CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id);
CREATE INDEX IF NOT EXISTS idx_detections_face ON detections(face_id);
CREATE INDEX IF NOT EXISTS idx_detections_upper_color ON detections(upper_color);
CREATE INDEX IF NOT EXISTS idx_detections_zone ON detections(zone_id);

-- ─────────────────────────────────────────────────
-- People Counts (time-series aggregates)
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS people_counts (
    id                      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id              TEXT REFERENCES detection_sessions(id) ON DELETE CASCADE,
    camera_id               TEXT NOT NULL,
    zone_id                 TEXT,
    timestamp               TIMESTAMPTZ NOT NULL,
    count                   INT NOT NULL DEFAULT 0,
    count_entering          INT NOT NULL DEFAULT 0,
    count_exiting           INT NOT NULL DEFAULT 0,
    count_by_upper_color    JSONB       -- {"red": 3, "blue": 5}
);
CREATE INDEX IF NOT EXISTS idx_counts_camera ON people_counts(camera_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_counts_zone ON people_counts(zone_id, timestamp DESC);

-- ─────────────────────────────────────────────────
-- Video Recordings
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS video_recordings (
    id                      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    camera_id               TEXT REFERENCES cameras(id) ON DELETE SET NULL,
    source_type             TEXT NOT NULL,  -- upload | stream_capture
    original_path           TEXT,
    compressed_path         TEXT,
    duration_seconds        FLOAT,
    original_size_bytes     BIGINT,
    compressed_size_bytes   BIGINT,
    compression_ratio       FLOAT,
    codec_original          TEXT,
    codec_compressed        TEXT,
    width                   INT,
    height                  INT,
    fps                     FLOAT,
    status                  TEXT NOT NULL DEFAULT 'uploaded',
    -- uploaded | queued | processing | completed | failed
    processing_task_id      TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_recordings_status ON video_recordings(status);
CREATE INDEX IF NOT EXISTS idx_recordings_camera ON video_recordings(camera_id);

-- ─────────────────────────────────────────────────
-- Webhooks
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS webhooks (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    secret      TEXT,
    events      JSONB NOT NULL,   -- ["person.detected", "face.recognized"]
    camera_ids  JSONB,            -- null = all cameras
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────
-- Face vector store table (pgvector direct search)
-- This mirrors what vector_store.py manages
-- ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS faces_vectors (
    id          TEXT PRIMARY KEY,
    vector      vector(512),
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_faces_vectors_embedding ON faces_vectors
    USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

-- Useful views
CREATE OR REPLACE VIEW v_camera_stats AS
SELECT
    c.id AS camera_id,
    c.name AS camera_name,
    c.location,
    c.is_active,
    COUNT(DISTINCT ds.id) AS total_sessions,
    SUM(ds.total_detections) AS total_detections,
    MAX(c.last_seen_at) AS last_active
FROM cameras c
LEFT JOIN detection_sessions ds ON ds.camera_id = c.id
GROUP BY c.id, c.name, c.location, c.is_active;

CREATE OR REPLACE VIEW v_hourly_counts AS
SELECT
    camera_id,
    zone_id,
    date_trunc('hour', timestamp) AS hour,
    AVG(count)::INT AS avg_count,
    MAX(count) AS peak_count,
    SUM(count_entering) AS total_entering,
    SUM(count_exiting) AS total_exiting
FROM people_counts
GROUP BY camera_id, zone_id, date_trunc('hour', timestamp);
