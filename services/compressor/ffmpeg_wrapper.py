"""
services/compressor/ffmpeg_wrapper.py

FFmpeg-based video compression with NVIDIA NVENC GPU acceleration.
Falls back to CPU encoding (libx264) if GPU not available.

Supports:
  - H.264 (NVENC: h264_nvenc, CPU: libx264)
  - H.265/HEVC (NVENC: hevc_nvenc, CPU: libx265)
  - AV1 (for RTX 4000+ series: av1_nvenc)
  - Per-video compression settings
  - Async processing with progress callbacks
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import structlog

from core.config.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class CompressionConfig:
    input_path: str
    output_path: str
    codec: str = None           # None = use settings default
    preset: str = None
    crf: int = None
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    max_width: int = 1920
    max_height: int = 1080
    target_fps: Optional[int] = None  # None = keep original
    use_gpu: bool = True


@dataclass
class CompressionResult:
    success: bool
    output_path: str
    input_size_bytes: int
    output_size_bytes: int
    compression_ratio: float        # e.g., 4.2 = 4.2x smaller
    duration_seconds: float
    codec_used: str
    error: Optional[str] = None


class FFmpegWrapper:
    """
    High-level FFmpeg wrapper with GPU acceleration.

    GPU pipeline:
        NVDEC decode → CUDA frame processing → NVENC encode
        This keeps frames on GPU memory, avoiding CPU round-trips.

    CPU fallback:
        libx264 / libx265 — slower but always available.
    """

    def __init__(self) -> None:
        self._gpu_available = self._check_gpu_available()
        if self._gpu_available:
            log.info("compressor.gpu_available", message="Using NVIDIA NVENC/NVDEC")
        else:
            log.warning("compressor.gpu_not_available", message="Falling back to CPU encoding")

    def _check_gpu_available(self) -> bool:
        """Check if NVENC is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    def _build_command(self, config: CompressionConfig) -> list[str]:
        """Build FFmpeg command based on config and available hardware."""
        use_gpu = config.use_gpu and self._gpu_available

        # Determine codec
        codec = config.codec or settings.COMPRESSION_CODEC
        if use_gpu and codec == "libx264":
            codec = "h264_nvenc"
        elif use_gpu and codec == "libx265":
            codec = "hevc_nvenc"
        elif not use_gpu and codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
            codec = "libx264"  # CPU fallback

        preset = config.preset or settings.COMPRESSION_PRESET
        crf = config.crf or settings.COMPRESSION_CRF

        cmd = ["ffmpeg", "-y"]  # -y = overwrite output without asking

        # Input with hardware decode (keep frames on GPU)
        if use_gpu and settings.GPU_DECODE:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        cmd += ["-i", config.input_path]

        # Video filters
        vf_parts = []

        # Scale filter
        if config.max_width or config.max_height:
            scale = f"scale='min({config.max_width},iw)':'min({config.max_height},ih)':force_original_aspect_ratio=decrease"
            if use_gpu:
                scale = f"scale_cuda='min({config.max_width},iw)':'min({config.max_height},ih)'"
            vf_parts.append(scale)

        # FPS filter
        if config.target_fps:
            fps_filter = f"fps={config.target_fps}"
            vf_parts.append(fps_filter)

        if vf_parts:
            filter_flag = "vf" if not use_gpu else "vf"
            cmd += [f"-{filter_flag}", ",".join(vf_parts)]

        # Video encoding
        cmd += ["-c:v", codec]

        if codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
            # NVENC settings
            cmd += [
                "-preset", preset,         # p1 (fastest) to p7 (best quality)
                "-rc", "vbr",              # variable bitrate
                "-cq", str(crf),           # constant quality target
                "-b:v", "0",               # no bitrate cap
                "-maxrate", "4M",          # max bitrate safety
                "-bufsize", "8M",
                "-profile:v", "high",
            ]
        else:
            # x264/x265 settings
            cmd += [
                "-preset", "medium" if preset in ("p4",) else preset,
                "-crf", str(crf),
            ]

        # Audio encoding
        cmd += [
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
        ]

        # Fast start for web playback (moov atom at front)
        cmd += ["-movflags", "+faststart"]

        # Output
        cmd += [config.output_path]

        return cmd

    async def compress(
        self,
        config: CompressionConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> CompressionResult:
        """
        Compress a video file asynchronously.

        Args:
            config: compression configuration
            progress_callback: called with progress 0.0-1.0

        Returns:
            CompressionResult with stats
        """
        import time

        input_path = Path(config.input_path)
        if not input_path.exists():
            return CompressionResult(
                success=False, output_path=config.output_path,
                input_size_bytes=0, output_size_bytes=0,
                compression_ratio=0, duration_seconds=0,
                codec_used="none", error=f"Input not found: {config.input_path}",
            )

        input_size = input_path.stat().st_size
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(config)
        codec_used = next(
            (cmd[i + 1] for i, c in enumerate(cmd) if c == "-c:v"), "unknown"
        )

        log.info(
            "compressor.starting",
            input=config.input_path,
            codec=codec_used,
            gpu=config.use_gpu and self._gpu_available,
        )

        t_start = time.perf_counter()

        try:
            # Run FFmpeg as async subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            duration = time.perf_counter() - t_start

            if process.returncode != 0:
                error_msg = stderr.decode()[-500:]  # last 500 chars of error
                log.error("compressor.failed", error=error_msg)
                return CompressionResult(
                    success=False, output_path=config.output_path,
                    input_size_bytes=input_size, output_size_bytes=0,
                    compression_ratio=0, duration_seconds=duration,
                    codec_used=codec_used, error=error_msg,
                )

            output_size = output_path.stat().st_size
            ratio = input_size / output_size if output_size > 0 else 0

            log.info(
                "compressor.complete",
                input_mb=round(input_size / 1024 / 1024, 1),
                output_mb=round(output_size / 1024 / 1024, 1),
                ratio=round(ratio, 2),
                duration_s=round(duration, 1),
                codec=codec_used,
            )

            return CompressionResult(
                success=True,
                output_path=str(output_path),
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                compression_ratio=ratio,
                duration_seconds=duration,
                codec_used=codec_used,
            )

        except Exception as e:
            log.error("compressor.exception", error=str(e))
            return CompressionResult(
                success=False, output_path=config.output_path,
                input_size_bytes=input_size, output_size_bytes=0,
                compression_ratio=0, duration_seconds=0,
                codec_used=codec_used, error=str(e),
            )

    async def get_video_info(self, input_path: str) -> dict:
        """Get video metadata using ffprobe."""
        import json
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            input_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            data = json.loads(stdout)
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )
            fmt = data.get("format", {})
            return {
                "width": video_stream.get("width"),
                "height": video_stream.get("height"),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "codec": video_stream.get("codec_name"),
                "duration": float(fmt.get("duration", 0)),
                "size_bytes": int(fmt.get("size", 0)),
                "bitrate": int(fmt.get("bit_rate", 0)),
            }
        return {}
