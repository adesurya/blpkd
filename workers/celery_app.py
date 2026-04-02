"""
workers/celery_app.py
Celery application configuration.
Queue routing ensures tasks go to appropriate workers (CPU vs GPU).
"""
from celery import Celery
from core.config.settings import settings

celery_app = Celery(
    "vision_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "workers.detection_tasks",
        "workers.face_tasks",
        "workers.compression_tasks",
        "workers.analytics_tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task routing — different queues for CPU and GPU workers
    task_routes={
        "workers.detection_tasks.*": {"queue": "gpu_tasks"},
        "workers.face_tasks.*": {"queue": "gpu_tasks"},
        "workers.compression_tasks.*": {"queue": "gpu_tasks"},
        "workers.analytics_tasks.*": {"queue": "cpu_tasks"},
    },

    # Task behavior
    task_acks_late=True,              # ack after completion (retry-safe)
    task_reject_on_worker_lost=True,  # re-queue if worker crashes
    task_soft_time_limit=300,         # 5 min soft limit (sends warning)
    task_time_limit=600,              # 10 min hard limit (kills task)

    # Result expiry
    result_expires=86400,             # 24 hours

    # Worker concurrency
    worker_prefetch_multiplier=1,     # don't prefetch GPU tasks

    # Beat schedule (periodic tasks)
    beat_schedule={
        "cluster-faces-every-30min": {
            "task": "workers.face_tasks.run_face_clustering",
            "schedule": 1800.0,       # every 30 minutes
        },
        "aggregate-analytics-every-5min": {
            "task": "workers.analytics_tasks.aggregate_counts",
            "schedule": 300.0,        # every 5 minutes
        },
    },
)
