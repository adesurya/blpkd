"""
workers/celery_app.py
Celery application configuration.

PENTING — Dua jenis worker:
  - CPU worker (worker-cpu): hanya jalankan analytics_tasks
  - GPU worker (worker-gpu): jalankan detection_tasks, face_tasks, compression_tasks

Masing-masing worker hanya load module yang dibutuhkan,
sehingga CPU worker tidak perlu import torch/ultralytics/insightface.
"""
from celery import Celery
from core.config.settings import settings

celery_app = Celery(
    "vision_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    # PENTING: jangan include semua task di sini!
    # Masing-masing worker menggunakan --include flag saat startup
    # (lihat docker-compose.yml command)
    # include di sini hanya untuk discovery beat schedule
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task routing — CPU vs GPU queue
    task_routes={
        "workers.detection_tasks.*":  {"queue": "gpu_tasks"},
        "workers.face_tasks.*":        {"queue": "gpu_tasks"},
        "workers.compression_tasks.*": {"queue": "gpu_tasks"},
        "workers.analytics_tasks.*":   {"queue": "cpu_tasks"},
    },

    # Task behavior
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=300,
    task_time_limit=600,

    # Result expiry
    result_expires=86400,

    # Worker prefetch
    worker_prefetch_multiplier=1,

    # Beat schedule (hanya untuk periodic tasks — tidak load GPU modules)
    beat_schedule={
        "cluster-faces-every-30min": {
            "task": "workers.analytics_tasks.run_face_clustering_trigger",
            "schedule": 1800.0,
        },
        "aggregate-analytics-every-5min": {
            "task": "workers.analytics_tasks.aggregate_counts",
            "schedule": 300.0,
        },
    },
)