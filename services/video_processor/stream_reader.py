"""
services/video_processor/stream_reader.py

Non-blocking stream reader untuk FastAPI.
PyAV container.demux() adalah blocking call — TIDAK boleh dijalankan
langsung di async event loop karena akan memblok seluruh API server.

Solusi: blocking I/O berjalan di background thread (threading.Thread).
Frame dikirim ke queue.Queue, lalu diambil oleh async generator
menggunakan loop.run_in_executor() agar tidak blocking.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)


class SourceType(str, Enum):
    RTSP   = "rtsp"
    M3U8   = "m3u8"
    FILE   = "file"
    WEBCAM = "webcam"


@dataclass
class VideoFrame:
    data:         np.ndarray
    frame_number: int
    timestamp:    float
    source_id:    str
    pts:          Optional[int] = None


@dataclass
class StreamConfig:
    source_id:       str
    source_url:      str
    source_type:     SourceType
    sample_rate:     int = 5
    max_width:       int = 1280
    max_height:      int = 720
    reconnect_delay: int = 5
    max_retries:     int = 10


class StreamReader:
    """
    Thread-based stream reader yang aman untuk async FastAPI.

    Background thread membaca frame (blocking).
    Async generator mengambil dari queue tanpa blocking event loop.
    """

    def __init__(self, config: StreamConfig) -> None:
        self._config  = config
        self._running = False
        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None

    def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncGenerator[VideoFrame, None]:
        """Async generator — yield VideoFrame tanpa blocking event loop."""
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_thread,
            name=f"stream-{self._config.source_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        loop = asyncio.get_event_loop()

        try:
            while True:
                try:
                    frame = await loop.run_in_executor(
                        None,
                        lambda: self._queue.get(timeout=2.0)
                    )
                except queue.Empty:
                    if not self._thread.is_alive() or not self._running:
                        break
                    continue

                if frame is None:  # sentinel
                    break
                yield frame
        finally:
            self._running = False

    def _reader_thread(self) -> None:
        """Background thread — semua blocking I/O di sini."""
        retry = 0
        while self._running and retry <= self._config.max_retries:
            try:
                self._read_blocking()
                if self._config.source_type == SourceType.FILE:
                    break
                if not self._running:
                    break
                retry += 1
                delay = min(self._config.reconnect_delay * (2 ** min(retry - 1, 4)), 60)
                log.warning("stream.reconnecting",
                            source_id=self._config.source_id,
                            attempt=retry, wait_sec=delay)
                time.sleep(delay)
            except Exception as exc:
                if not self._running:
                    break
                retry += 1
                log.error("stream.read_error",
                          source_id=self._config.source_id,
                          error=str(exc), attempt=retry)
                if self._config.source_type == SourceType.FILE:
                    break
                time.sleep(self._config.reconnect_delay)

        try:
            self._queue.put(None, timeout=5.0)
        except queue.Full:
            pass

    def _read_blocking(self) -> None:
        """Buka stream dan baca frame — blocking, aman di thread."""
        import av

        url = self._config.source_url
        options: dict[str, str] = {}

        if self._config.source_type == SourceType.RTSP:
            options = {
                "rtsp_transport": "tcp",
                "stimeout":       "5000000",
                "buffer_size":    "1048576",
            }
        elif self._config.source_type == SourceType.M3U8:
            options = {
                "protocol_whitelist": "file,http,https,tcp,tls,crypto",
            }

        try:
            container = av.open(url, options=options, timeout=15.0)
        except Exception as exc:
            log.error("stream.open_failed", url=url, error=str(exc))
            raise

        try:
            vstream = container.streams.video[0]
            vstream.thread_type = "AUTO"
            fps = float(vstream.average_rate) if vstream.average_rate else 25.0
            log.info("stream.ready",
                     source_id=self._config.source_id,
                     fps=round(fps, 1),
                     resolution=f"{vstream.width}x{vstream.height}")

            frame_idx = 0
            for packet in container.demux(vstream):
                if not self._running:
                    return
                try:
                    for av_frame in packet.decode():
                        if not self._running:
                            return
                        frame_idx += 1
                        if frame_idx % self._config.sample_rate != 0:
                            continue

                        bgr = av_frame.to_ndarray(format="bgr24")
                        bgr = self._maybe_resize(bgr)
                        ts = (
                            float(av_frame.pts * av_frame.time_base)
                            if av_frame.pts is not None else time.time()
                        )
                        try:
                            self._queue.put(
                                VideoFrame(data=bgr, frame_number=frame_idx,
                                           timestamp=ts,
                                           source_id=self._config.source_id,
                                           pts=av_frame.pts),
                                timeout=1.0
                            )
                        except queue.Full:
                            log.debug("stream.frame_dropped", frame=frame_idx)
                except av.AVError:
                    continue
        finally:
            container.close()

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._config.max_width and h <= self._config.max_height:
            return frame
        scale = min(self._config.max_width / w, self._config.max_height / h)
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)