"""
services/api/middleware/logging_middleware.py
Request logging and correlation ID injection.
"""
from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every request with:
    - Unique correlation ID (X-Request-ID header)
    - Method, path, status, duration
    - Client IP
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        # Bind correlation ID to all log entries in this request
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t_start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            log.error(
                "request.error",
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - t_start) * 1000
            status_code = getattr(response, "status_code", 500)

            # Skip health check logging to reduce noise
            if request.url.path not in ("/health", "/metrics"):
                log.info(
                    "request.complete",
                    method=request.method,
                    path=request.url.path,
                    status=status_code,
                    duration_ms=round(duration_ms, 1),
                    client_ip=request.client.host if request.client else "unknown",
                )

            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id
        return response
