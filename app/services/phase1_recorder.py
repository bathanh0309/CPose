"""
CPose — app/services/phase1_recorder.py
Phase 1: Real-time RTSP ingestion, YOLOv8n person detection, MP4 clip recording.

Architecture:
  RecorderManager         → manages N CameraWorker threads
  CameraWorker (Thread)   → reads frames, infers, writes clips
  FrameBuffer (deque)     → rolling pre-detection buffer
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger("[Phase1]")

# ─── Tuning constants ────────────────────────────────────────────────────────
PRE_BUFFER_SEC    = 3       # seconds of footage kept before detection
POST_BUFFER_SEC   = 3       # seconds to keep recording after last detection
INFERENCE_EVERY   = 5       # run YOLO every N frames (performance trade-off)
MIN_CLIP_DURATION = 2.0     # minimum seconds for a clip to be saved
PERSON_CLASS_ID   = 0       # COCO class index for "person"
CONF_THRESHOLD_P1 = 0.35    # confidence threshold Phase 1
RECONNECT_DELAY   = 5       # seconds to wait before reconnecting a failed stream


class CameraWorker(threading.Thread):
    """
    One thread per camera.
    Reads frames from RTSP, runs YOLOv8n periodically,
    and records MP4 clips when persons are detected.
    """

    def __init__(
        self,
        cam_id: str,
        url: str,
        output_dir: Path,
        model_path: Path,
        width: int | None,
        height: int | None,
        storage_limit_getter,
        socketio_emit,
    ):
        super().__init__(daemon=True, name=f"cam-{cam_id}")
        self.cam_id          = cam_id
        self.url             = url
        self.output_dir      = output_dir
        self.model_path      = model_path
        self.req_width       = width
        self.req_height      = height
        self._storage_limit  = storage_limit_getter   # callable → float GB
        self._emit           = socketio_emit
        self._stop_evt       = threading.Event()

        # State
        self.status          = "connecting"
        self.fps_actual      = 0.0
        self.resolution      = (0, 0)
        self._model          = None

    # ──────────────────────────────────────────────────────────────────────
    def run(self):
        self._load_model()
        while not self._stop_evt.is_set():
            try:
                self._stream_loop()
            except Exception as exc:
                logger.error("cam-%s stream error: %s", self.cam_id, exc)
                self.status = "error"
                self._emit("camera_status", {
                    "cam_id": self.cam_id, "status": "error", "message": str(exc)
                })
            if not self._stop_evt.is_set():
                logger.info("cam-%s reconnecting in %ds…", self.cam_id, RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)

    def stop(self):
        self._stop_evt.set()

    # ──────────────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            logger.info("cam-%s: YOLOv8n loaded from %s", self.cam_id, self.model_path)
        except Exception as exc:
            logger.error("cam-%s: failed to load model: %s", self.cam_id, exc)
            self._emit("error", {"source": f"cam-{self.cam_id}", "message": str(exc)})

    # ──────────────────────────────────────────────────────────────────────
    def _stream_loop(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if self.req_width and self.req_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)

        if not cap.isOpened():
            raise ConnectionError(f"Cannot open RTSP: {self.url}")

        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_actual  = stream_fps
        self.resolution  = (w, h)
        self.status      = "streaming"

        self._emit("camera_status", {
            "cam_id": self.cam_id,
            "status": "streaming",
            "fps": round(stream_fps, 2),
            "resolution": f"{w}x{h}",
        })
        logger.info("cam-%s connected  %dx%d @ %.1f fps", self.cam_id, w, h, stream_fps)

        pre_buf_len     = int(PRE_BUFFER_SEC * stream_fps)
        post_buf_frames = int(POST_BUFFER_SEC * stream_fps)
        frame_buffer    = deque(maxlen=pre_buf_len)

        writer          = None
        clip_path       = None
        no_person_cnt   = 0
        frame_idx       = 0
        clip_start_time = None

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("cam-%s: frame read failed", self.cam_id)
                    break

                frame_buffer.append(frame.copy())
                frame_idx += 1

                # ── Inference ──────────────────────────────────────────────
                person_detected = False
                if self._model and frame_idx % INFERENCE_EVERY == 0:
                    try:
                        results = self._model.predict(
                            frame,
                            classes=[PERSON_CLASS_ID],
                            conf=CONF_THRESHOLD_P1,
                            verbose=False,
                        )
                        count = 0
                        confidence_max = 0.0
                        for result in results:
                            for box in result.boxes:
                                if int(box.cls[0]) != PERSON_CLASS_ID:
                                    continue
                                count += 1
                                confidence_max = max(confidence_max, float(box.conf[0]))
                        if count > 0:
                            person_detected = True
                            self._emit("detection_event", {
                                "cam_id": self.cam_id,
                                "timestamp": datetime.now().isoformat(),
                                "person_count": count,
                                "confidence_max": round(confidence_max, 3),
                            })
                    except Exception as exc:
                        logger.error("cam-%s inference error: %s", self.cam_id, exc)

                # ── Recording FSM ──────────────────────────────────────────
                if person_detected:
                    no_person_cnt = 0
                    if writer is None:
                        # Enforce storage before starting a new clip
                        self._enforce_storage()
                        clip_path, writer = self._open_writer(w, h, stream_fps)
                        clip_start_time = time.time()
                        # Flush pre-buffer
                        for buffered in frame_buffer:
                            writer.write(buffered)
                        logger.info("cam-%s: clip started → %s", self.cam_id, clip_path.name)
                    writer.write(frame)

                elif writer is not None:
                    writer.write(frame)
                    no_person_cnt += 1
                    if no_person_cnt >= post_buf_frames:
                        duration = time.time() - clip_start_time
                        writer.release()
                        writer = None
                        if duration >= MIN_CLIP_DURATION:
                            size_mb = clip_path.stat().st_size / 1e6
                            self._emit("clip_saved", {
                                "filename": clip_path.name,
                                "size_mb": round(size_mb, 2),
                                "duration_s": round(duration, 1),
                                "cam_id": self.cam_id,
                            })
                            logger.info(
                                "cam-%s: clip saved %s (%.1fs, %.1fMB)",
                                self.cam_id, clip_path.name, duration, size_mb,
                            )
                        else:
                            clip_path.unlink(missing_ok=True)
                            logger.debug("cam-%s: clip too short, discarded", self.cam_id)
                        clip_path = None
                        no_person_cnt = 0

        finally:
            if writer is not None:
                writer.release()
            cap.release()
            self.status = "disconnected"

    # ──────────────────────────────────────────────────────────────────────
    def _open_writer(self, w: int, h: int, fps: float):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_cam{self.cam_id}.mp4"
        path = self.output_dir / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        return path, writer

    def _enforce_storage(self):
        from app.utils.file_handler import FileHandler
        limit_gb = self._storage_limit()
        FileHandler().enforce_storage_limit(self.output_dir, limit_gb)


# ═══════════════════════════════════════════════════════════════════════════
class RecorderManager:
    """Singleton-style manager for all CameraWorker threads."""

    def __init__(self):
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self._storage_limit_gb = 10.0

    # ──────────────────────────────────────────────────────────────────────
    def start(
        self,
        cameras: list[dict[str, Any]],
        storage_limit_gb: float,
        output_dir: Path,
        model_path: Path,
    ):
        from app import socketio

        def _emit(event: str, data: dict):
            socketio.emit(event, data)

        self._storage_limit_gb = storage_limit_gb

        with self._lock:
            for cam in cameras:
                cam_id = str(cam["cam_id"]).zfill(2)
                if cam_id in self._workers:
                    logger.warning("cam-%s already running, skipping", cam_id)
                    continue
                worker = CameraWorker(
                    cam_id=cam_id,
                    url=cam["url"],
                    output_dir=output_dir,
                    model_path=model_path,
                    width=cam.get("width"),
                    height=cam.get("height"),
                    storage_limit_getter=lambda: self._storage_limit_gb,
                    socketio_emit=_emit,
                )
                self._workers[cam_id] = worker
                worker.start()
                logger.info("cam-%s worker started", cam_id)

    def stop(self):
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=2.0)
        logger.info("All camera workers stopped")

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._workers)

    def set_storage_limit(self, limit_gb: float):
        self._storage_limit_gb = limit_gb
        logger.info("Storage limit updated to %.1f GB", limit_gb)

    def status(self) -> dict:
        with self._lock:
            cameras = []
            for cam_id, w in self._workers.items():
                cameras.append({
                    "cam_id": cam_id,
                    "status": w.status,
                    "fps": round(w.fps_actual, 2),
                    "resolution": f"{w.resolution[0]}x{w.resolution[1]}",
                })
            return {
                "running": bool(self._workers),
                "cameras": cameras,
                "storage_limit_gb": self._storage_limit_gb,
            }
