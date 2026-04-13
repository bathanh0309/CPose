"""
CPose — app/services/phase1_recorder.py
Phase 1: RTSP ingestion + Storage-Aware Person-Triggered Recording.

State machine per camera:
  IDLE      → no person detected
  ARMED     → person seen, confirming (need N consecutive detections)
  RECORDING → confirmed person, actively writing clip
  POST_ROLL → person gone, recording tail before closing clip
  SAVED     → clip finalized (transient, resets to IDLE)

Architecture:
  RecorderManager         → manages N CameraWorker threads, exposes snapshots
  CameraWorker (Thread)   → frame loop + YOLO + FSM + VideoWriter
  FrameBuffer (deque)     → pre-roll circular buffer
"""
from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger("[Phase1]")

# ─── Default tuning constants (all overridable via config) ────────────────────
PERSON_CONF_THRESHOLD      = 0.65    # minimum confidence to count as valid person
TRIGGER_MIN_CONSECUTIVE    = 3       # consecutive inference frames with person → confirm
PRE_ROLL_SECONDS           = 3       # circular buffer length before trigger
POST_ROLL_SECONDS          = 5       # keep recording after last person sighting
REARM_COOLDOWN_SECONDS     = 5       # min gap between clips (merge window)
MIN_CLIP_SECONDS           = 3.0     # discard clip shorter than this
MAX_CLIP_SECONDS           = 300.0   # force-close clip at this duration
MIN_BOX_AREA_RATIO         = 0.0015  # bbox must be > 0.15% of frame area
INFERENCE_EVERY            = 3       # run YOLO every N frames
RECONNECT_DELAY            = 5       # seconds before reconnect attempt
JPEG_QUALITY               = 75      # snapshot compression quality
PERSON_CLASS_ID            = 0       # COCO class index for "person"


class RecState(Enum):
    IDLE      = "idle"
    ARMED     = "armed"
    RECORDING = "recording"
    POST_ROLL = "post_roll"


class CameraWorker(threading.Thread):
    """One daemon thread per RTSP camera.  Reads frames, infers, FSM-records."""

    def __init__(
        self,
        cam_id: str,
        url: str,
        label: str,
        output_dir: Path,
        model_path: Path,
        width: int | None,
        height: int | None,
        storage_limit_getter: Callable[[], float],
        socketio_emit: Callable[[str, dict], None],
        config: dict | None = None,
    ):
        super().__init__(daemon=True, name=f"cam-{cam_id}")
        self.cam_id = cam_id
        self.url = url
        self.label = label or f"cam{cam_id}"
        self.output_dir = output_dir
        self.model_path = model_path
        self.req_width = width
        self.req_height = height
        self._storage_limit = storage_limit_getter
        self._emit = socketio_emit
        self._stop_evt = threading.Event()

        cfg = config or {}
        self._conf_threshold    = float(cfg.get("person_conf_threshold",   PERSON_CONF_THRESHOLD))
        self._min_consecutive   = int(cfg.get("trigger_min_consecutive",   TRIGGER_MIN_CONSECUTIVE))
        self._pre_roll_sec      = float(cfg.get("pre_roll_seconds",        PRE_ROLL_SECONDS))
        self._post_roll_sec     = float(cfg.get("post_roll_seconds",       POST_ROLL_SECONDS))
        self._cooldown_sec      = float(cfg.get("rearm_cooldown_seconds",  REARM_COOLDOWN_SECONDS))
        self._min_clip_sec      = float(cfg.get("min_clip_seconds",        MIN_CLIP_SECONDS))
        self._max_clip_sec      = float(cfg.get("max_clip_seconds",        MAX_CLIP_SECONDS))
        self._min_box_area      = float(cfg.get("min_box_area_ratio",      MIN_BOX_AREA_RATIO))
        self._inference_every   = int(cfg.get("inference_every",           INFERENCE_EVERY))

        # Public status (read by RecorderManager)
        self.status: str = "connecting"
        self.fps_actual: float = 0.0
        self.resolution: tuple[int, int] = (0, 0)
        self.rec_state: RecState = RecState.IDLE
        self.person_detected: bool = False
        self.current_clip_duration: float = 0.0
        self.clip_count: int = 0
        self.detect_count: int = 0

        # Snapshot (latest JPEG bytes, thread-safe)
        self._snap_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._model = None

    # ── Public API ──────────────────────────────────────────────────────────
    def run(self) -> None:
        self._load_model()
        while not self._stop_evt.is_set():
            try:
                self._stream_loop()
            except Exception as exc:
                logger.error("cam-%s stream error: %s", self.cam_id, exc)
                self.status = "error"
                self.person_detected = False
                self.rec_state = RecState.IDLE
                self._emit("camera_status", self._status_payload("error", message=str(exc)))
                self._set_snapshot(None)
            if not self._stop_evt.is_set():
                logger.info("cam-%s reconnecting in %ds…", self.cam_id, RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)

    def stop(self) -> None:
        self._stop_evt.set()

    def get_snapshot(self) -> bytes | None:
        with self._snap_lock:
            return self._latest_jpeg

    # ── Internal ────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            logger.info("cam-%s: model loaded (%s)", self.cam_id, self.model_path.name)
        except Exception as exc:
            logger.error("cam-%s: model load failed: %s", self.cam_id, exc)
            self._emit("error", {"source": f"cam-{self.cam_id}", "message": str(exc)})

    def _stream_loop(self) -> None:  # noqa: C901
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if self.req_width and self.req_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        if not cap.isOpened():
            raise ConnectionError(f"Cannot open RTSP: {self.url}")

        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = w * h
        self.fps_actual = stream_fps
        self.resolution = (w, h)
        self.status = "online"

        self._emit("camera_status", self._status_payload("online"))
        logger.info("cam-%s online  %dx%d @ %.1f fps", self.cam_id, w, h, stream_fps)

        pre_buf_size  = max(1, int(self._pre_roll_sec * stream_fps))
        post_frames   = max(1, int(self._post_roll_sec * stream_fps))
        pre_buffer    = deque(maxlen=pre_buf_size)

        # FSM state
        fsm           = RecState.IDLE
        consecutive   = 0      # consecutive infer-frames with valid person (ARMED counter)
        no_person_cnt = 0      # frames without person (POST_ROLL counter)
        writer: cv2.VideoWriter | None = None
        clip_path: Path | None = None
        clip_start: float = 0.0
        last_clip_end: float = 0.0   # for cooldown
        frame_idx = 0

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("cam-%s: read failed", self.cam_id)
                    break

                self._update_snapshot(frame)
                pre_buffer.append(frame.copy())
                frame_idx += 1

                # ── Inference (every N frames) ────────────────────────────
                person_now = False
                if self._model and frame_idx % self._inference_every == 0:
                    person_now, conf_max = self._detect_person(frame, frame_area)
                    if person_now:
                        self.detect_count += 1
                        self._emit("detection_event", {
                            "cam_id": self.cam_id,
                            "timestamp": datetime.now().isoformat(),
                            "confidence_max": round(conf_max, 3),
                            "person_count": 1,
                        })

                self.person_detected = person_now

                # ── FSM transitions ───────────────────────────────────────
                if fsm == RecState.IDLE:
                    if person_now:
                        # Check cooldown (merge-window)
                        if time.time() - last_clip_end < self._cooldown_sec:
                            pass  # still in cooldown
                        else:
                            fsm = RecState.ARMED
                            consecutive = 1
                            logger.info("cam-%s: IDLE→ARMED (person seen)", self.cam_id)
                            self._log("person detected, trigger armed")

                elif fsm == RecState.ARMED:
                    if person_now:
                        consecutive += 1
                        if consecutive >= self._min_consecutive:
                            # Confirmed — start recording
                            self._enforce_storage()
                            clip_path, writer = self._open_writer(w, h, stream_fps)
                            clip_start = time.time()
                            # Flush pre-roll buffer
                            for buffered in pre_buffer:
                                writer.write(buffered)
                            fsm = RecState.RECORDING
                            no_person_cnt = 0
                            logger.info("cam-%s: ARMED→RECORDING → %s", self.cam_id, clip_path.name)
                            self._log(f"recording started: {clip_path.name}")
                            self._emit("camera_status", self._status_payload("recording"))
                    else:
                        # Not confirmed, disarm
                        fsm = RecState.IDLE
                        consecutive = 0
                        self._log("trigger disarmed (not enough confirmation frames)")

                elif fsm == RecState.RECORDING:
                    if writer:
                        writer.write(frame)
                        self.current_clip_duration = time.time() - clip_start

                    if person_now:
                        no_person_cnt = 0
                        # Extend: log every 30 seconds
                        if frame_idx % max(1, int(30 * stream_fps)) == 0:
                            self._log("recording extended (person still present)")
                    else:
                        no_person_cnt += 1
                        fsm = RecState.POST_ROLL
                        logger.info("cam-%s: RECORDING→POST_ROLL", self.cam_id)

                    # Force-close at max duration
                    if self.current_clip_duration >= self._max_clip_sec:
                        logger.info("cam-%s: max duration reached, closing clip", self.cam_id)
                        clip_path = self._close_clip(writer, clip_path, clip_start)
                        writer = None
                        clip_path = None
                        last_clip_end = time.time()
                        fsm = RecState.IDLE
                        self._emit("camera_status", self._status_payload("online"))

                elif fsm == RecState.POST_ROLL:
                    if writer:
                        writer.write(frame)

                    if person_now:
                        # Person returned — back to recording
                        no_person_cnt = 0
                        fsm = RecState.RECORDING
                        logger.info("cam-%s: POST_ROLL→RECORDING (person returned)", self.cam_id)
                        self._log("recording extended (person returned in post-roll)")
                    else:
                        no_person_cnt += 1
                        if no_person_cnt >= post_frames:
                            # Close clip
                            clip_path = self._close_clip(writer, clip_path, clip_start)
                            writer = None
                            clip_path = None
                            last_clip_end = time.time()
                            fsm = RecState.IDLE
                            consecutive = 0
                            no_person_cnt = 0
                            self.current_clip_duration = 0.0
                            self._emit("camera_status", self._status_payload("online"))

                self.rec_state = fsm

        finally:
            if writer is not None:
                self._close_clip(writer, clip_path, clip_start)
            cap.release()
            self.status = "offline"
            self.person_detected = False
            self.rec_state = RecState.IDLE
            self._set_snapshot(None)

    def _detect_person(self, frame, frame_area: int) -> tuple[bool, float]:
        """Run YOLO, return (person_confirmed, max_confidence)."""
        try:
            results = self._model.predict(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=self._conf_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error("cam-%s inference error: %s", self.cam_id, exc)
            return False, 0.0

        best_conf = 0.0
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                if conf < self._conf_threshold:
                    continue
                # Check minimum box area
                x1, y1, x2, y2 = box.xyxy[0]
                box_area = float((x2 - x1) * (y2 - y1))
                if frame_area > 0 and (box_area / frame_area) < self._min_box_area:
                    logger.debug("cam-%s: small box skipped (area ratio %.4f)",
                                 self.cam_id, box_area / frame_area)
                    continue
                best_conf = max(best_conf, conf)

        return best_conf > 0, best_conf

    def _close_clip(self, writer: cv2.VideoWriter, clip_path: Path | None, clip_start: float) -> Path | None:
        writer.release()
        if clip_path is None or not clip_path.exists():
            return None

        duration = time.time() - clip_start
        if duration < self._min_clip_sec:
            clip_path.unlink(missing_ok=True)
            self._log(f"clip discarded (too short: {duration:.1f}s < {self._min_clip_sec}s)")
            logger.info("cam-%s: clip too short (%.1fs), discarded", self.cam_id, duration)
            return None

        size_mb = clip_path.stat().st_size / 1e6
        self.clip_count += 1
        self._emit("clip_saved", {
            "filename": clip_path.name,
            "size_mb": round(size_mb, 2),
            "duration_s": round(duration, 1),
            "cam_id": self.cam_id,
        })
        self._log(f"clip saved: {clip_path.name} ({duration:.1f}s, {size_mb:.1f} MB)")
        logger.info("cam-%s: clip saved %s (%.1fs, %.1fMB)", self.cam_id, clip_path.name, duration, size_mb)
        return clip_path

    def _open_writer(self, w: int, h: int, fps: float) -> tuple[Path, cv2.VideoWriter]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_cam{self.cam_id}.mp4"
        path = self.output_dir / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        return path, writer

    def _update_snapshot(self, frame) -> None:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            self._set_snapshot(buf.tobytes())
        except Exception:
            pass

    def _set_snapshot(self, data: bytes | None) -> None:
        with self._snap_lock:
            self._latest_jpeg = data

    def _enforce_storage(self) -> None:
        from app.utils.file_handler import FileHandler
        limit_gb = self._storage_limit()
        FileHandler().enforce_storage_limit(self.output_dir, limit_gb)

    def _log(self, message: str) -> None:
        self._emit("rec_log", {"cam_id": self.cam_id, "message": message,
                               "timestamp": datetime.now().isoformat()})

    def _status_payload(self, status: str, **extra) -> dict:
        payload = {
            "cam_id": self.cam_id,
            "label": self.label,
            "status": status,
            "fps": round(self.fps_actual, 2),
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "person_detected": self.person_detected,
            "recording": self.rec_state in (RecState.RECORDING, RecState.POST_ROLL),
            "rec_state": self.rec_state.value,
            "clip_duration": round(self.current_clip_duration, 1),
            "clip_count": self.clip_count,
            "detect_count": self.detect_count,
        }
        payload.update(extra)
        return payload


# ═══════════════════════════════════════════════════════════════════════════════
class RecorderManager:
    """Manages all CameraWorker threads.  Thread-safe singleton-style."""

    def __init__(self) -> None:
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self._storage_limit_gb = 10.0
        self._config: dict = {}

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def start(
        self,
        cameras: list[dict[str, Any]],
        storage_limit_gb: float,
        output_dir: Path,
        model_path: Path,
        config: dict | None = None,
    ) -> None:
        from app import socketio

        def _emit(event: str, data: dict) -> None:
            socketio.emit(event, data)

        self._storage_limit_gb = storage_limit_gb
        self._config = config or {}

        with self._lock:
            for cam in cameras:
                cam_id = str(cam["cam_id"]).zfill(2)
                if cam_id in self._workers:
                    logger.warning("cam-%s already running, skipping", cam_id)
                    continue
                worker = CameraWorker(
                    cam_id=cam_id,
                    url=cam["url"],
                    label=cam.get("label", f"cam{cam_id}"),
                    output_dir=output_dir,
                    model_path=model_path,
                    width=cam.get("width"),
                    height=cam.get("height"),
                    storage_limit_getter=lambda: self._storage_limit_gb,
                    socketio_emit=_emit,
                    config=self._config,
                )
                self._workers[cam_id] = worker
                worker.start()
                logger.info("cam-%s worker started (url: %s)", cam_id, cam["url"])

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=3.0)
        logger.info("All camera workers stopped")

    def is_running(self) -> bool:
        with self._lock:
            return any(w.is_alive() for w in self._workers.values())

    def set_storage_limit(self, limit_gb: float) -> None:
        self._storage_limit_gb = limit_gb

    # ── Snapshot ────────────────────────────────────────────────────────────
    def get_snapshot(self, cam_id: str) -> bytes | None:
        cam_id = str(cam_id).zfill(2)
        with self._lock:
            worker = self._workers.get(cam_id)
        return worker.get_snapshot() if worker else None

    # ── Status ──────────────────────────────────────────────────────────────
    def status(self) -> dict:
        with self._lock:
            cameras = []
            for cam_id, w in self._workers.items():
                cameras.append({
                    "cam_id": cam_id,
                    "label": w.label,
                    "status": w.status,
                    "fps": round(w.fps_actual, 2),
                    "resolution": f"{w.resolution[0]}x{w.resolution[1]}",
                    "person_detected": w.person_detected,
                    "recording": w.rec_state in (RecState.RECORDING, RecState.POST_ROLL),
                    "rec_state": w.rec_state.value,
                    "clip_duration": round(w.current_clip_duration, 1),
                    "clip_count": w.clip_count,
                    "detect_count": w.detect_count,
                })
            return {
                "running": self.is_running(),
                "cameras": cameras,
                "storage_limit_gb": self._storage_limit_gb,
            }

    def camera_status_list(self) -> list[dict]:
        return self.status()["cameras"]
