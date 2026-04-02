"""
services/video_processor/stream_reader.py

Video stream ingestion for realtime (m3u8/RTSP) and recorded video files.
Handles reconnection, frame sampling, and queuing frames for processing.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import av
import cv2
import numpy as np
import structlog

from core.config.settings import settings

log = structlog.get_logger(__name__)


class SourceType(str, Enum):
    RTSP = "rtsp"
    M3U8 = "m3u8"           # HLS stream (live or VoD)
    FILE = "file"            # recorded video file
    WEBCAM = "webcam"        # local webcam (dev/testing)


@dataclass
class VideoFrame:
    data: np.ndarray        # BGR numpy array
    frame_number: int
    timestamp: float         # seconds from start
    source_id: str
    pts: Optional[int] = None  # presentation timestamp


@dataclass
class StreamConfig:
    source_id: str           # camera ID or upload ID
    source_url: str
    source_type: SourceType
    sample_rate: int = 5     # process every Nth frame (1 = every frame)
    max_width: int = 1920
    max_height: int = 1080
    reconnect_delay: int = 5  # seconds between reconnect attempts
    max_retries: int = 10


class StreamReader:
    """
    Unified video reader for all source types.

    Features:
    - Automatic reconnection for live streams
    - Frame sampling (configurable)
    - Frame resizing for consistent processing
    - Async generator interface for clean downstream consumption
    - Error recovery with exponential backoff

    Usage:
        reader = StreamReader(config)
        async for frame in reader.frames():
            # frame is a VideoFrame
            result = await detector.process(frame)
    """

    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._is_running = False
        self._frame_count = 0
        self._retry_count = 0

    async def frames(self) -> AsyncGenerator[VideoFrame, None]:
        """
        Async generator that yields VideoFrame objects.
        Automatically handles reconnection for live streams.
        """
        self._is_running = True
        log.info(
            "stream.starting",
            source_id=self._config.source_id,
            url=self._config.source_url,
            type=self._config.source_type,
        )

        while self._is_running and self._retry_count <= self._config.max_retries:
            try:
                async for frame in self._read_source():
                    if not self._is_running:
                        return
                    yield frame

                # If we reach here on a live stream, connection was lost
                if self._config.source_type in (SourceType.RTSP, SourceType.M3U8):
                    self._retry_count += 1
                    delay = min(
                        self._config.reconnect_delay * (2 ** min(self._retry_count, 4)),
                        60,
                    )
                    log.warning(
                        "stream.reconnecting",
                        source_id=self._config.source_id,
                        retry=self._retry_count,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    # File source — done
                    log.info("stream.file_completed", source_id=self._config.source_id)
                    return

            except Exception as e:
                log.error(
                    "stream.error",
                    source_id=self._config.source_id,
                    error=str(e),
                    retry=self._retry_count,
                )
                if self._config.source_type == SourceType.FILE:
                    raise
                self._retry_count += 1
                await asyncio.sleep(self._config.reconnect_delay)

        log.error("stream.max_retries_reached", source_id=self._config.source_id)

    async def _read_source(self) -> AsyncGenerator[VideoFrame, None]:
        """Internal: read frames from the actual source using PyAV."""
        url = self._config.source_url

        # OpenCV options for RTSP
        options = {}
        if self._config.source_type == SourceType.RTSP:
            options = {
                "rtsp_transport": "tcp",
                "stimeout": "5000000",   # 5 second socket timeout
                "buffer_size": "1024000",
            }
        elif self._config.source_type == SourceType.M3U8:
            options = {
                "protocol_whitelist": "file,http,https,tcp,tls,crypto",
            }

        try:
            # Use PyAV for better stream compatibility
            container = av.open(
                url,
                options=options,
                timeout=10.0,
            )
        except av.AVError as e:
            log.error("stream.open_failed", url=url, error=str(e))
            raise

        try:
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"   # multi-threaded decode

            fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
            self._retry_count = 0  # reset on successful open

            log.info(
                "stream.opened",
                source_id=self._config.source_id,
                fps=fps,
                resolution=f"{video_stream.width}x{video_stream.height}",
            )

            frame_index = 0
            for packet in container.demux(video_stream):
                if not self._is_running:
                    return

                for av_frame in packet.decode():
                    if not self._is_running:
                        return

                    frame_index += 1

                    # Frame sampling: skip frames based on sample_rate
                    if frame_index % self._config.sample_rate != 0:
                        continue

                    # Convert to BGR numpy array (OpenCV format)
                    bgr_frame = av_frame.to_ndarray(format="bgr24")

                    # Resize if needed
                    bgr_frame = self._resize_frame(bgr_frame)

                    timestamp = float(av_frame.pts * av_frame.time_base) if av_frame.pts else time.time()

                    yield VideoFrame(
                        data=bgr_frame,
                        frame_number=frame_index,
                        timestamp=timestamp,
                        source_id=self._config.source_id,
                        pts=av_frame.pts,
                    )

        finally:
            container.close()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if it exceeds max dimensions."""
        h, w = frame.shape[:2]
        max_w = self._config.max_width
        max_h = self._config.max_height

        if w <= max_w and h <= max_h:
            return frame

        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def stop(self) -> None:
        """Signal the stream to stop gracefully."""
        self._is_running = False
        log.info("stream.stopping", source_id=self._config.source_id)


class StreamManager:
    """
    Manages multiple concurrent stream readers.
    Keeps track of active streams, their status, and metrics.
    """

    def __init__(self) -> None:
        self._streams: dict[str, tuple[StreamReader, asyncio.Task]] = {}

    async def start_stream(
        self,
        config: StreamConfig,
        on_frame: Callable[[VideoFrame], None],
    ) -> None:
        """Start processing a stream in the background."""
        if config.source_id in self._streams:
            log.warning("stream.already_running", source_id=config.source_id)
            return

        reader = StreamReader(config)

        async def _consume():
            async for frame in reader.frames():
                try:
                    await on_frame(frame) if asyncio.iscoroutinefunction(on_frame) else on_frame(frame)
                except Exception as e:
                    log.error("stream.frame_handler_error", error=str(e))

        task = asyncio.create_task(_consume(), name=f"stream-{config.source_id}")
        self._streams[config.source_id] = (reader, task)
        log.info("stream.started", source_id=config.source_id)

    async def stop_stream(self, source_id: str) -> None:
        """Stop a running stream."""
        if source_id not in self._streams:
            return
        reader, task = self._streams.pop(source_id)
        reader.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        log.info("stream.stopped", source_id=source_id)

    def get_active_streams(self) -> list[str]:
        return list(self._streams.keys())

    async def stop_all(self) -> None:
        for source_id in list(self._streams.keys()):
            await self.stop_stream(source_id)
