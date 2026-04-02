"""
core/config/logging.py
Structured logging setup using structlog.

PrintLoggerFactory tidak punya attribute .name, jadi tidak bisa
pakai add_logger_name. Gunakan stdlib.add_log_level saja,
atau switch ke stdlib logging factory.
"""
from __future__ import annotations

import logging
import sys

import structlog
from core.config.settings import settings


def configure_logging() -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # DIHAPUS: structlog.stdlib.add_logger_name
        # → hanya bekerja dengan stdlib LoggerFactory, bukan PrintLoggerFactory
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=False),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if settings.DEBUG else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)