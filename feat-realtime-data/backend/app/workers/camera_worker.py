# Worker chính đọc RTSP, detect, cập nhật preview và kích hoạt ghi hình.
import os
import asyncio
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings
from app.schemas import CameraStatusUpdate, EventCreate, EventType
from app.services.detector_service import detector_service
from app.services.event_bus import event_bus
from app.services.processing_service import processing_service
from app.services.preview_streamer import preview_streamer
from app.services.recorder_service import recorder_service
from app.services.rtsp_manager import rtsp_manager
from app.utils.resources_loader import CameraConfig

# Force RTSP over TCP for OpenCV FFmpeg backend to prevent packet loss.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

logger = logging.getLogger(__name__)
_settings = get_settings()


class CameraWorker:
    def __init__(self, config: CameraConfig, loop: asyncio.AbstractEventLoop) -> None:
        self._config = config
        self._loop = loop
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        # Separate bounded queues so preview and AI never block ingest.
        self._detect_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._preview_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        self._person_present = False
        self._last_detection_time = 0.0
        self._post_roll_secs = _settings.clip_seconds_after
        self._metrics_interval_secs = 1.0

        self._source_fps = 0.0
        self._detect_fps = 0.0
        self._preview_fps = 0.0
        self._decoder_errors = 0

    def start(self) -> None:
        ingest = threading.Thread(
            target=self._ingest_loop,
            name=f"ingest-{self._config.id}",
            daemon=True,
        )
        detect = threading.Thread(
            target=self._detect_loop,
            name=f"detect-{self._config.id}",
            daemon=True,
        )
        preview = threading.Thread(
            target=self._preview_loop,
            name=f"preview-{self._config.id}",
            daemon=True,
        )

        self._threads.extend([ingest, detect, preview])
        for thread in self._threads:
            thread.start()
        logger.info("CameraWorker started: %s", self._config.id)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=3)
        logger.info("CameraWorker stopped: %s", self._config.id)

    def _ingest_loop(self) -> None:
        retry_delay = 2.0
        while not self._stop_event.is_set():
            cap = None
            stream_opened = False
            reconnect_delay = retry_delay

            try:
                cap = cv2.VideoCapture(self._config.rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    reconnect_delay = retry_delay
                    logger.warning(
                        "[%s] Cannot open stream. Retry in %.1fs",
                        self._config.id,
                        retry_delay,
                    )
                    self._dispatch_status(online=False)
                    retry_delay = min(retry_delay * 1.5, 30.0)
                    continue

                retry_delay = 2.0
                stream_opened = True

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                cap_fps = self._normalize_capture_fps(cap.get(cv2.CAP_PROP_FPS))

                self._source_fps = cap_fps
                self._detect_fps = 0.0
                self._preview_fps = 0.0

                self._dispatch_status(online=True)
                logger.info(
                    "[%s] Connected %sx%s @ %.1f fps",
                    self._config.id,
                    width,
                    height,
                    cap_fps,
                )

                fps_frame_count = 0
                fps_window_start = time.monotonic()

                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        self._decoder_errors += 1
                        logger.warning(
                            "[%s] Stream read failed (decoder_errors=%d). Reconnecting...",
                            self._config.id,
                            self._decoder_errors,
                        )
                        break

                    now = time.monotonic()
                    fps_frame_count += 1
                    if now - fps_window_start >= self._metrics_interval_secs:
                        self._source_fps = fps_frame_count / (now - fps_window_start)
                        fps_frame_count = 0
                        fps_window_start = now
                        self._dispatch_metrics()

                    self._push_latest(self._detect_q, frame)

                    cam_state = rtsp_manager.get(self._config.id)
                    if cam_state and cam_state.preview_enabled:
                        self._push_latest(self._preview_q, frame)

                    if recorder_service.is_recording(self._config.id):
                        recorder_service.write_frame(self._config.id, frame)
            except Exception:
                logger.exception("[%s] Unexpected ingest error. Reconnecting...", self._config.id)
            finally:
                if cap is not None:
                    cap.release()

                if stream_opened:
                    clip_path = self._stop_recording()
                    if clip_path:
                        logger.info("[%s] Finalized clip before reconnect: %s", self._config.id, clip_path)
                    self._person_present = False
                    self._dispatch_status(online=False)

                if not self._stop_event.is_set():
                    self._stop_event.wait(reconnect_delay)

    def _detect_loop(self) -> None:
        fps_frame_count = 0
        fps_window_start = time.monotonic()

        while not self._stop_event.is_set():
            try:
                frame = self._detect_q.get(timeout=1.0)
            except queue.Empty:
                continue

            detected = detector_service.detect(frame)

            now = time.monotonic()
            fps_frame_count += 1
            if now - fps_window_start >= self._metrics_interval_secs:
                self._detect_fps = fps_frame_count / (now - fps_window_start)
                fps_frame_count = 0
                fps_window_start = now
                self._dispatch_metrics()

            if detected:
                self._last_detection_time = now
                if not self._person_present:
                    self._person_present = True
                    self._on_person_detected(frame)
            elif self._person_present:
                person_gone = now - self._last_detection_time > self._post_roll_secs
                if person_gone:
                    self._person_present = False
                    self._on_person_left()

    def _preview_loop(self) -> None:
        prev_w = _settings.preview_width
        prev_h = _settings.preview_height
        target_delay = 1.0 / max(_settings.preview_fps, 1.0)

        fps_frame_count = 0
        fps_window_start = time.monotonic()

        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            try:
                frame = self._preview_q.get(timeout=1.0)
            except queue.Empty:
                continue

            cam_state = rtsp_manager.get(self._config.id)
            if not cam_state or not cam_state.preview_enabled:
                if self._preview_fps != 0.0:
                    self._preview_fps = 0.0
                    self._dispatch_metrics()
                time.sleep(0.2)
                continue

            preview_frame = cv2.resize(frame, (prev_w, prev_h))
            success, jpeg = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if not success:
                continue

            preview_streamer.push_frame(self._config.id, jpeg.tobytes())

            now = time.monotonic()
            fps_frame_count += 1
            if now - fps_window_start >= self._metrics_interval_secs:
                self._preview_fps = fps_frame_count / (now - fps_window_start)
                fps_frame_count = 0
                fps_window_start = now
                self._dispatch_metrics()

            elapsed = time.monotonic() - loop_started
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

    def _on_person_detected(self, frame: np.ndarray) -> None:
        logger.info("[%s] Person detected", self._config.id)
        self._dispatch_event(EventType.person_detected)
        self._start_recording(frame)

    def _on_person_left(self) -> None:
        logger.info("[%s] Person left", self._config.id)
        self._dispatch_event(EventType.person_left)
        self._stop_recording()

    def _start_recording(self, frame: np.ndarray) -> Optional[str]:
        height, width = frame.shape[:2]
        out_path = recorder_service.start_clip(
            self._config.id,
            self._config.name,
            width,
            height,
            max(self._source_fps, 1.0),
            [],
        )
        if out_path:
            self._dispatch_event(EventType.recording_started, str(out_path))
            return str(out_path)
        return None

    def _stop_recording(self) -> Optional[str]:
        clip_path = recorder_service.stop_clip(self._config.id)
        if clip_path:
            self._dispatch_event(EventType.recording_stopped, clip_path)
            processing_service.enqueue_clip(
                raw_path=clip_path,
                camera_id=self._config.id,
                camera_name=self._config.name,
            )
        return clip_path

    def _dispatch_event(self, evt: EventType, clip_path: Optional[str] = None) -> None:
        asyncio.run_coroutine_threadsafe(
            event_bus.publish(
                EventCreate(
                    event=evt,
                    camera_id=self._config.id,
                    cam_name=self._config.name,
                    clip_path=clip_path,
                )
            ),
            self._loop,
        )

    def _dispatch_status(self, online: bool) -> None:
        last_seen_at = None
        if online:
            last_seen_at = datetime.utcnow().isoformat() + "Z"
            rtsp_manager.mark_online(
                self._config.id,
                source_fps=self._source_fps,
                detect_fps=self._detect_fps,
                preview_fps=self._preview_fps,
            )
        else:
            self._source_fps = 0.0
            self._detect_fps = 0.0
            self._preview_fps = 0.0
            rtsp_manager.on_offline(self._config.id)

        update = CameraStatusUpdate(
            id=self._config.id,
            online=online,
            source_fps=self._source_fps,
            detect_fps=self._detect_fps,
            preview_fps=self._preview_fps,
            decoder_errors=self._decoder_errors,
            last_seen_at=last_seen_at,
        )
        asyncio.run_coroutine_threadsafe(event_bus.publish_camera_status(update), self._loop)

    def _dispatch_metrics(self) -> None:
        last_seen_at = datetime.utcnow().isoformat() + "Z"
        rtsp_manager.update_metrics(
            self._config.id,
            self._source_fps,
            self._detect_fps,
            self._preview_fps,
            self._decoder_errors,
        )
        update = CameraStatusUpdate(
            id=self._config.id,
            online=True,
            source_fps=self._source_fps,
            detect_fps=self._detect_fps,
            preview_fps=self._preview_fps,
            decoder_errors=self._decoder_errors,
            last_seen_at=last_seen_at,
        )
        asyncio.run_coroutine_threadsafe(event_bus.publish_camera_status(update), self._loop)

    @staticmethod
    def _push_latest(target_queue: queue.Queue[np.ndarray], frame: np.ndarray) -> None:
        try:
            target_queue.put_nowait(frame)
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                target_queue.put_nowait(frame)
            except queue.Full:
                pass

    @staticmethod
    def _normalize_capture_fps(capture_fps: float) -> float:
        if capture_fps and 1.0 <= capture_fps <= 120.0:
            return float(capture_fps)
        return 25.0
