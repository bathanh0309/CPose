# Tiện ích kiểm tra FFmpeg và dựng lệnh encode video đầu ra.
"""
FFmpeg utility helpers.
Checks that ffmpeg is available on PATH at startup and provides
reusable command builders.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    found = shutil.which("ffmpeg") is not None
    if not found:
        logger.warning(
            "ffmpeg not found on PATH. Recording will be disabled. "
            "Install it with: sudo apt install ffmpeg  (or brew install ffmpeg)"
        )
    return found


def ffmpeg_pipe_cmd(
    width: int,
    height: int,
    fps: float,
    output_path: str,
    crf: int = 23,
    preset: str = "ultrafast",
) -> list[str]:
    """
    Build an FFmpeg command that reads raw BGR24 frames from stdin
    and encodes to H.264 MP4.
    """
    return [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",   # optimize for web playback
        output_path,
    ]
