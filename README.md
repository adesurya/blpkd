# 🎯 Vision Platform — AI-Powered Video Analytics Backend

> Production-ready backend for AI Vision, Face Recognition, People Counting & Video Analytics.
> Designed to scale from MVP (single GPU) → Growth (cluster) → Enterprise (multi-tenant K8s).

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quick Start (MVP)](#quick-start-mvp)
5. [Configuration: CCTV & Video Sources](#configuration-cctv--video-sources)
6. [REST API Documentation](#rest-api-documentation)
7. [Custom Model Training](#custom-model-training)
8. [Scaling Guide](#scaling-guide)
9. [Monitoring](#monitoring)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT / DASHBOARD                          │
│                    REST API  |  WebSocket  |  Webhook               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                          FastAPI Gateway                            │
│               Auth (JWT) | Rate Limit | Request Routing             │
└──┬──────────────┬──────────────┬──────────────┬─────────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
[Stream API] [Video Upload] [Face API] [Analytics API]
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘
                        │ Redis Queue (MVP)
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                      WORKER POOL (Celery)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │   Detector  │  │FaceEngine   │  │ Compressor  │               │
│  │  YOLOv8 +  │  │InsightFace  │  │ FFmpeg+     │               │
│  │  ByteTrack  │  │buffalo_l    │  │ NVENC       │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
│  ┌─────────────┐  ┌─────────────┐                                 │
│  │  Attribute  │  │  Person     │                                  │
│  │  Analyzer   │  │  ReID       │                                  │
│  │  OpenCV HSV │  │  BotSORT    │                                  │
│  └─────────────┘  └─────────────┘                                 │
└───────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐  ┌──────────────────┐  ┌────────────────┐
│  PostgreSQL  │  │   MinIO          │  │  Redis         │
│  + pgvector  │  │  Object Storage  │  │  Cache/Queue   │
│  (metadata,  │  │  (video, frames, │  │                │
│   embeddings)│  │   face crops)    │  │                │
└──────────────┘  └──────────────────┘  └────────────────┘
        │
        ▼
┌──────────────────────────────┐
│   Prometheus + Grafana       │
│   Structured Logs (structlog)│
└──────────────────────────────┘
```

### Scale Path (No Code Rewrite Required)
```
MVP:        Redis Queue → RabbitMQ → Apache Kafka
MVP:        pgvector   → pgvector  → Milvus
MVP:        MinIO(1)   → MinIO(3+) → MinIO/S3
MVP:        Compose    → Compose   → Kubernetes (Helm)
```

---

## Tech Stack

| Category | MVP | Growth | Enterprise |
|----------|-----|--------|------------|
| Language | Python 3.11+ | Python 3.11+ | Python 3.11+ |
| API | FastAPI | FastAPI | FastAPI |
| Task Queue | Redis + Celery | RabbitMQ + Celery | Kafka + Faust |
| Detection | YOLOv8n/s | YOLOv8m/l | Custom fine-tuned |
| Face Recog. | InsightFace | InsightFace | InsightFace + cluster |
| Vector DB | pgvector | pgvector | Milvus |
| Object Store | MinIO (1-node) | MinIO (3-node) | MinIO / S3 |
| Database | PostgreSQL 16 | PostgreSQL HA | PostgreSQL cluster |
| GPU | RTX 3060+ | Tesla T4/A10 | A100 multi-GPU |
| Deploy | Docker Compose | Docker Compose | Kubernetes |

---

## Project Structure

```
vision-platform/
├── services/
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py             # App entry point
│   │   ├── routers/            # API route handlers
│   │   │   ├── streams.py      # Stream management
│   │   │   ├── videos.py       # Video upload & processing
│   │   │   ├── faces.py        # Face management
│   │   │   ├── analytics.py    # Analytics & counting
│   │   │   └── webhooks.py     # Webhook management
│   │   ├── middleware/         # Auth, rate limit, CORS
│   │   └── dependencies.py     # DI: DB, storage, queue
│   │
│   ├── video_processor/        # Stream & video ingestion
│   │   ├── stream_reader.py    # m3u8/RTSP stream consumer
│   │   ├── frame_extractor.py  # Frame sampling logic
│   │   └── hls_manager.py      # HLS stream management
│   │
│   ├── detector/               # YOLOv8 people detection
│   │   ├── yolo_detector.py    # YOLO inference wrapper
│   │   ├── tracker.py          # ByteTrack/BotSORT tracker
│   │   └── counter.py          # People counting logic
│   │
│   ├── face_engine/            # InsightFace recognition
│   │   ├── detector.py         # Face detection (SCRFD)
│   │   ├── recognizer.py       # Face recognition (ArcFace)
│   │   ├── clusterer.py        # DBSCAN clustering
│   │   └── gallery.py          # Face gallery management
│   │
│   ├── attribute_analyzer/     # Clothing & attribute analysis
│   │   ├── color_detector.py   # OpenCV HSV color detection
│   │   ├── clothing_classifier.py  # YOLOv8 custom classifier
│   │   └── activity_recognizer.py  # Activity recognition
│   │
│   └── compressor/             # Video compression service
│       ├── ffmpeg_wrapper.py   # FFmpeg subprocess wrapper
│       ├── gpu_encoder.py      # NVENC GPU encoding
│       └── pipeline.py         # Compression pipeline
│
├── workers/                    # Celery worker tasks
│   ├── __init__.py
│   ├── celery_app.py           # Celery configuration
│   ├── detection_tasks.py      # Detection job tasks
│   ├── face_tasks.py           # Face processing tasks
│   ├── compression_tasks.py    # Video compression tasks
│   └── analytics_tasks.py     # Analytics aggregation tasks
│
├── core/                       # Shared abstractions
│   ├── abstractions/
│   │   ├── queue.py            # Queue interface (swap Redis→Kafka)
│   │   ├── storage.py          # Storage interface (swap MinIO→S3)
│   │   └── vector_store.py     # Vector DB interface (pgvector→Milvus)
│   ├── config/
│   │   ├── settings.py         # Pydantic Settings
│   │   └── logging.py          # structlog configuration
│   └── models/
│       ├── domain.py           # Domain models (Pydantic)
│       └── database.py         # SQLAlchemy ORM models
│
├── storage/
│   └── migrations/             # Alembic migrations
│
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.worker
│   │   └── Dockerfile.gpu_worker
│   ├── nginx/
│   │   └── nginx.conf
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/dashboards/
│
├── docs/
│   ├── api/                    # API documentation
│   ├── architecture/           # Architecture diagrams
│   └── training/               # Model training guides
│
├── scripts/
│   ├── training/
│   │   ├── train_clothing_classifier.py
│   │   ├── train_activity_recognizer.py
│   │   └── prepare_dataset.py
│   └── setup/
│       ├── init_db.py
│       └── seed_data.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env.example
├── docker-compose.yml          # MVP full stack
├── docker-compose.gpu.yml      # GPU worker override
├── docker-compose.scale.yml    # Growth scale override
└── pyproject.toml
```
