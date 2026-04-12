# Lưu frame JPEG mới nhất của mỗi camera để stream MJPEG ra trình duyệt.
"""
Preview Streamer: maintains a per-camera ring buffer of the latest JPEG frame
for MJPEG streaming to the browser.

Design principles:
- Preview is COMPLETELY decoupled from detection and recording.
- Workers push frames here at the preview FPS rate (configurable, default 15).
- API route /api/cameras/{id}/preview reads from here and streams MJPEG.
- If a camera goes offline, the streamer serves the last known frame (or blank).
- Frame rate is throttled here to protect CPU; source capture is unaffected.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


def _blank_frame(width: int = 320, height: int = 240) -> bytes:
    """Returns a JPEG-encoded black frame as bytes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


_BLANK = _blank_frame(_settings.preview_width, _settings.preview_height)


class PreviewStreamer:
    """
    Thread-safe (asyncio-safe) per-camera JPEG frame store.
    Workers call push_frame(); MJPEG endpoint calls latest_frame().
    """

    def __init__(self) -> None:
        # camera_id → latest JPEG bytes
        self._frames: Dict[str, bytes] = {}
        # camera_id → asyncio.Event signalling a new frame arrived
        self._events: Dict[str, asyncio.Event] = {}

    def _get_event(self, camera_id: str) -> asyncio.Event:
        if camera_id not in self._events:
            self._events[camera_id] = asyncio.Event()
        return self._events[camera_id]

    def push_frame(self, camera_id: str, jpeg_bytes: bytes) -> None:
        """Called by CameraWorker (sync context) to update the latest frame."""
        self._frames[camera_id] = jpeg_bytes
        # Signal any waiting async consumers
        evt = self._events.get(camera_id)
        if evt:
            # Schedule set() safely from a non-async context if needed
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(evt.set)
            except RuntimeError:
                pass  # no running loop — will be picked up on next poll

    def latest_frame(self, camera_id: str) -> bytes:
        """Returns the most recent JPEG frame, or a blank if none yet."""
        return self._frames.get(camera_id, _BLANK)

    def clear_camera(self, camera_id: str) -> None:
        self._frames.pop(camera_id, None)
        self._events.pop(camera_id, None)

    def reset(self) -> None:
        self._frames.clear()
        self._events.clear()

    async def frame_generator(self, camera_id: str):
        """
        Async generator that yields MJPEG-boundary-wrapped JPEG frames.
        Used directly by the FastAPI streaming response.
        Yields at the configured preview FPS rate.
        """
        min_interval = 1.0 / _settings.preview_fps
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

        while True:
            t0 = time.monotonic()
            frame = self.latest_frame(camera_id)
            yield boundary + frame + b"\r\n"

            elapsed = time.monotonic() - t0
            sleep = max(0.0, min_interval - elapsed)
            await asyncio.sleep(sleep)


# Singleton
preview_streamer = PreviewStreamer()
