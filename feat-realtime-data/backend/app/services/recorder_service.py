# Quản lý recorder ghi clip MP4 và hàng đợi frame cho từng camera.
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from app.config import get_settings
from app.services.file_namer import clip_path

logger = logging.getLogger(__name__)
_settings = get_settings()

# Max recording resolution (width). Frames wider than this are downscaled
# before encoding so the CPU encoder can keep up with high-res cameras.
_MAX_RECORD_WIDTH = 1280


class ClipRecorder:
    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        self.output_path = output_path
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_thread: Optional[threading.Thread] = None
        # Scale down resolution if needed so encoder can keep up
        if width > _MAX_RECORD_WIDTH:
            scale = _MAX_RECORD_WIDTH / width
            self._width = _MAX_RECORD_WIDTH
            self._height = int(height * scale) & ~1  # keep even for codec
        else:
            self._width = width
            self._height = height
        # Queue sized for ~2 seconds; kept small to avoid memory bloat
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max(int(fps * 2), 30))
        self._fps = max(fps, 1.0)
        self._frame_count = 0
        self._dropped_frames = 0
        self._stop_requested = False

    def start(self, preroll_frames: list[np.ndarray]) -> bool:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use XVID (AVI container trick) or mp4v — XVID is faster on CPU
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self._fps,
            (self._width, self._height),
        )
        if not self._writer.isOpened():
            logger.error("Unable to open MP4 writer for %s", self.output_path)
            self._writer = None
            return False

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"recorder-{self.output_path.stem}",
            daemon=True,
        )
        self._writer_thread.start()

        logger.info(
            "Recording started: %s (encode res: %dx%d @ %.0ffps)",
            self.output_path.name, self._width, self._height, self._fps,
        )
        for frame in preroll_frames:
            self.write_frame(frame)
        return True

    def write_frame(self, frame: np.ndarray) -> None:
        if self._stop_requested or self._writer is None:
            return

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped_frames += 1
            if self._dropped_frames == 1 or self._dropped_frames % 30 == 0:
                logger.warning(
                    "Recorder backlog for %s: dropped %d frame(s); encoder cannot keep up with incoming frames",
                    self.output_path.name,
                    self._dropped_frames,
                )
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop_requested = True
        self._queue.put(self._sentinel)
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5)

        if self._writer:
            try:
                self._writer.release()
            except Exception as exc:
                logger.error("Failed to close writer for %s: %s", self.output_path.name, exc)
            logger.info(
                "Recording stopped: %s (%d frames, dropped=%d)",
                self.output_path.name,
                self._frame_count,
                self._dropped_frames,
            )
            self._writer = None

    @property
    def is_running(self) -> bool:
        return self._writer is not None and not self._stop_requested

    @property
    def _sentinel(self) -> None:
        return None

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to the target encoding resolution if it doesn't match."""
        h, w = frame.shape[:2]
        if w != self._width or h != self._height:
            return cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _writer_loop(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is self._sentinel:
                break

            if self._writer is None:
                continue

            try:
                frame = self._resize_if_needed(frame)
                self._writer.write(frame)
                self._frame_count += 1
            except Exception as exc:
                logger.error("MP4 write error for %s: %s", self.output_path.name, exc)
                break


class RecorderService:
    def __init__(self) -> None:
        self._active: Dict[str, ClipRecorder] = {}

    def start_clip(
        self,
        camera_id: str,
        _camera_name: str,
        width: int,
        height: int,
        fps: float,
        preroll: list[np.ndarray],
    ) -> Optional[Path]:
        from datetime import datetime
        if camera_id in self._active:
            return None

        ts = datetime.utcnow()
        out_path = clip_path(_settings.recordings_dir, camera_id, ts)
        recorder = ClipRecorder(out_path, width, height, fps)
        if recorder.start(preroll):
            self._active[camera_id] = recorder
            return out_path
        return None

    def write_frame(self, camera_id: str, frame: np.ndarray) -> None:
        rec = self._active.get(camera_id)
        if rec and rec.is_running:
            rec.write_frame(frame)

    def stop_clip(self, camera_id: str) -> Optional[str]:
        rec = self._active.pop(camera_id, None)
        if rec:
            rec.stop()
            return str(rec.output_path)
        return None

    def is_recording(self, camera_id: str) -> bool:
        return camera_id in self._active

    def stop_all(self) -> None:
        camera_ids = list(self._active.keys())
        for camera_id in camera_ids:
            self.stop_clip(camera_id)

recorder_service = RecorderService()
