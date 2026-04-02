"""
workers/celery_app.py
Celery configuration.

Semua tasks ada di satu file: workers/detection_tasks.py
- GPU tasks: process_stream_frame, process_video_file, run_face_clustering, compress_video
- CPU tasks: aggregate_counts, send_webhook, run_face_clustering_trigger
"""
from celery import Celery
from core.config.settings import settings

celery_app = Celery(
    "vision_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Routing: semua tasks ada di detection_tasks.py
    task_routes={
        "workers.detection_tasks.process_stream_frame":  {"queue": "gpu_tasks"},
        "workers.detection_tasks.process_video_file":    {"queue": "gpu_tasks"},
        "workers.detection_tasks.compress_video":        {"queue": "gpu_tasks"},
        "workers.detection_tasks.run_face_clustering":   {"queue": "gpu_tasks"},
        "workers.detection_tasks.aggregate_counts":      {"queue": "cpu_tasks"},
        "workers.detection_tasks.send_webhook":          {"queue": "cpu_tasks"},
        "workers.detection_tasks.run_face_clustering_trigger": {"queue": "cpu_tasks"},
    },

    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=300,
    task_time_limit=600,
    result_expires=86400,
    worker_prefetch_multiplier=1,

    # Beat schedule — hanya trigger ringan, tidak load GPU modules
    beat_schedule={
        "cluster-faces-every-30min": {
            "task": "workers.detection_tasks.run_face_clustering_trigger",
            "schedule": 1800.0,
        },
        "aggregate-analytics-every-5min": {
            "task": "workers.detection_tasks.aggregate_counts",
            "schedule": 300.0,
        },
    },
)