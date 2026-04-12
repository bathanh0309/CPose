# Worker detect tách riêng để xử lý bất đồng bộ khi tải suy luận tăng cao.
"""
DetectionWorker: optional standalone worker for decoupling detection
from the main CameraWorker loop.

Current architecture: detection is called inline inside CameraWorker._run().
This file provides a queue-based alternative for cases where:
  - Detection is slower than capture rate (GPU-heavy model)
  - You want to run detection on a separate thread/process
  - You want to scale detection independently of capture

Usage: instantiate and call start(). Push frames via submit().
The worker calls back into CameraWorker's detection state machine
via a provided callback function.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

from app.services.detector_service import detector_service

logger = logging.getLogger(__name__)


class DetectionWorker:
    """
    Runs detector_service.detect() in a dedicated thread.
    Frames are submitted via a bounded queue; old frames are dropped
    if the queue is full (detection always operates on the freshest frame).
    """

    def __init__(
        self,
        camera_id: str,
        on_detected: Callable[[], None],
        on_clear: Callable[[], None],
        queue_size: int = 2,
    ) -> None:
        self._camera_id = camera_id
        self._on_detected = on_detected
        self._on_clear = on_clear
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"detect-{self._camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Unblock the queue.get() call
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=3)

    def submit(self, frame: np.ndarray) -> None:
        """
        Submit a frame for detection. If the queue is full, drop the oldest
        frame and submit the new one — we always want the freshest frame.
        """
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                break

            try:
                detected = detector_service.detect(frame)
                if detected:
                    self._on_detected()
                else:
                    self._on_clear()
            except Exception as e:
                logger.error("[%s] Detection error: %s", self._camera_id, e)
