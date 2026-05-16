from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class SequentialFileCamera:
    """Sequential local video reader for tracking-sensitive web sessions."""

    def __init__(self, source: str, loop: bool = True):
        self.source = str(source)
        self.loop = bool(loop)
        self.cap = cv2.VideoCapture(self.source)
        self.frame_idx = 0
        self.last_frame_ts = 0.0
        self._error: str | None = None
        self._opened = bool(self.cap and self.cap.isOpened())
        if not self._opened:
            self._error = f"OpenCV cannot open local file: {Path(self.source).name}"
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0) if self._opened else 25.0
        if fps <= 1 or fps > 240:
            fps = 25.0
        self.fps = fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640) if self._opened else 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480) if self._opened else 480

    def start(self) -> "SequentialFileCamera":
        return self

    def wait_opened(self, timeout: float = 0.0) -> bool:
        return self._opened

    def read(self, copy: bool = True) -> np.ndarray | None:
        if not self.cap or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if (not ok or frame is None) and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            ok, frame = self.cap.read()
        if not ok or frame is None:
            self._error = "End of local video."
            return None
        self.frame_idx += 1
        return frame.copy() if copy else frame

    def get_meta(self) -> dict:
        return {"fps": self.fps, "width": self.width, "height": self.height}

    def age(self) -> float:
        return 0.0

    def error(self) -> str | None:
        return self._error

    def release(self, join_timeout: float = 0.0) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
