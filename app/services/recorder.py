from __future__ import annotations

import io
import logging
import os
import threading
import time
import queue
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.utils.runtime_config import get_runtime_section

logger = logging.getLogger("[Phase1]")

_PHASE1_CFG = get_runtime_section("phase1")

# Handle Pydantic model conversion to dict
try:
    if hasattr(_PHASE1_CFG, "model_dump"):
        _phase1_dict = _PHASE1_CFG.model_dump()
    else:
        _phase1_dict = dict(_PHASE1_CFG)
except (AttributeError, TypeError):
    _phase1_dict = {}

# ─── Default tuning constants (all overridable via config) ────────────────────
PERSON_CONF_THRESHOLD      = float(_phase1_dict.get("person_conf_threshold", 0.65))
TRIGGER_MIN_CONSECUTIVE    = int(_phase1_dict.get("trigger_min_consecutive", 3))
PRE_ROLL_SECONDS           = float(_phase1_dict.get("pre_roll_seconds", 3))
POST_ROLL_SECONDS          = float(_phase1_dict.get("post_roll_seconds", 5))
REARM_COOLDOWN_SECONDS     = float(_phase1_dict.get("rearm_cooldown_seconds", 5))
MIN_CLIP_SECONDS           = float(_phase1_dict.get("min_clip_seconds", 3.0))
MAX_CLIP_SECONDS           = float(_phase1_dict.get("max_clip_seconds", 300.0))
MIN_BOX_AREA_RATIO         = float(_phase1_dict.get("min_box_area_ratio", 0.0015))
INFERENCE_EVERY            = int(_phase1_dict.get("inference_every", 3))
RECONNECT_DELAY            = int(_phase1_dict.get("reconnect_delay_seconds", 5))
JPEG_QUALITY               = int(_phase1_dict.get("jpeg_quality", 75))
PERSON_CLASS_ID            = int(_phase1_dict.get("person_class_id", 0))
SNAPSHOT_FPS               = float(_phase1_dict.get("snapshot_fps", 6.0))
SNAPSHOT_ACTIVE_TTL_S      = float(_phase1_dict.get("snapshot_active_ttl_s", 10.0))
RTSP_TRANSPORT_DEFAULT     = str(_phase1_dict.get("rtsp_transport", "tcp")).strip().lower()
FFMPEG_CAPTURE_OPTIONS     = str(_phase1_dict.get("ffmpeg_capture_options", "")).strip()


class RecState(Enum):
    IDLE      = "idle"
    ARMED     = "armed"
    RECORDING = "recording"
    POST_ROLL = "post_roll"


class CameraWorker(threading.Thread):
    """
    One daemon thread per RTSP camera.
    Uses three internal threads for optimal performance:
    1. Ingest: Reads frames from RTSP and handles VideoWriter.
    2. Detect: Runs YOLO inference on a separate thread to avoid blocking ingest.
    3. Preview: Encodes snapshots for the UI at a controlled FPS.
    """

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
        on_clip_ready: Callable[[Path, str], None] | None = None,
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
        self._on_clip_ready = on_clip_ready

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

        # Separate bounded queues so preview and AI never block ingest.
        # Maxsize 1 ensures we always process the LATEST frame.
        self._detect_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._preview_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

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
        self._latest_jpeg_ts: float = 0.0
        
        # Internal state for FSM
        self._writer: cv2.VideoWriter | None = None
        self._clip_path: Path | None = None
        self._clip_start_time: float = 0.0
        self._last_clip_end_time: float = 0.0
        self._pre_buffer = deque()

        # Snapshot tuning
        self._snapshot_fps = float(cfg.get("snapshot_fps", SNAPSHOT_FPS))
        self._snapshot_active_ttl = float(cfg.get("snapshot_active_ttl_s", SNAPSHOT_ACTIVE_TTL_S))
        self._snapshot_last_encoded_at: float = 0.0
        self._snapshot_last_requested_at: float = 0.0

        # RTSP tuning
        default_rtsp_transport = os.environ.get("CPOSE_RTSP_TRANSPORT", RTSP_TRANSPORT_DEFAULT)
        self._rtsp_transport = str(cfg.get("rtsp_transport", default_rtsp_transport)).strip().lower()
        self._ffmpeg_capture_options = str(cfg.get("ffmpeg_capture_options", FFMPEG_CAPTURE_OPTIONS)).strip()
        self._model = None

    # ── Public API ──────────────────────────────────────────────────────────
    def run(self) -> None:
        import queue  # Ensure queue is available in the thread scope
        self._load_model()
        
        # Start helper threads
        detect_thread = threading.Thread(target=self._detect_loop, name=f"detect-{self.cam_id}", daemon=True)
        preview_thread = threading.Thread(target=self._preview_loop, name=f"preview-{self.cam_id}", daemon=True)
        
        detect_thread.start()
        preview_thread.start()
        
        retry_delay = 2.0
        while not self._stop_evt.is_set():
            try:
                self._stream_ingest_loop(retry_delay)
                # If loop finishes normally (e.g. read fail), reset retry
                retry_delay = 2.0
            except Exception as exc:
                logger.error("cam-%s stream error: %s", self.cam_id, exc)
                self.status = "error"
                self.person_detected = False
                self.rec_state = RecState.IDLE
                self._emit("camera_status", self._status_payload("error", message=str(exc)))
                self._set_snapshot(None)
                # Exponential backoff
                retry_delay = min(retry_delay * 1.5, 30.0)

            if not self._stop_evt.is_set():
                logger.info("cam-%s reconnecting in %.1fs…", self.cam_id, retry_delay)
                time.sleep(retry_delay)

    def stop(self) -> None:
        self._stop_evt.set()

    def get_snapshot(self) -> bytes | None:
        self._snapshot_last_requested_at = time.time()
        # Note: encoding happens on the preview_thread, we just return the latest bytes
        with self._snap_lock:
            return self._latest_jpeg

    # ── Internal Loops ──────────────────────────────────────────────────────
    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            logger.info("cam-%s: model loaded (%s)", self.cam_id, self.model_path.name)
        except Exception as exc:
            logger.error("cam-%s: model load failed: %s", self.cam_id, exc)
            self._emit("error", {"source": f"cam-{self.cam_id}", "message": str(exc)})

    def _stream_ingest_loop(self, retry_delay: float) -> None:  # noqa: C901
        """Reads frames and manages VideoWriter based on current FSM state."""
        if self._ffmpeg_capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._ffmpeg_capture_options
        elif self._rtsp_transport in ("tcp", "udp") and not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self._rtsp_transport}"

        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
            
        if self.req_width and self.req_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
            
        if not cap.isOpened():
            raise ConnectionError(f"Cannot open RTSP: {self.url}")

        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_actual = stream_fps
        self.resolution = (w, h)
        self.status = "online"

        self._emit("rec_status", self._status_payload("online"))
        logger.info("cam-%s online  %dx%d @ %.1f fps", self.cam_id, w, h, stream_fps)

        pre_buf_size = max(1, int(self._pre_roll_sec * stream_fps))
        self._pre_buffer = deque(maxlen=pre_buf_size)
        
        frame_idx = 0
        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("cam-%s: read failed (frame is null or ret=False)", self.cam_id)
                    break

                frame_idx += 1
                
                # Push to helper queues (latest only)
                self._push_latest(self._detect_q, frame)
                
                # Only push to preview if active (requested within TTL)
                if (time.time() - self._snapshot_last_requested_at) < self._snapshot_active_ttl:
                    self._push_latest(self._preview_q, frame)

                # Manage Pre-Roll Buffer
                if self.rec_state == RecState.IDLE or self.rec_state == RecState.ARMED:
                    self._pre_buffer.append(frame.copy())

                # Manage Writing
                if self.rec_state in (RecState.RECORDING, RecState.POST_ROLL):
                    if self._writer:
                        self._writer.write(frame)
                        self.current_clip_duration = time.time() - self._clip_start_time
                        
                        # Safety: Force-close at max duration
                        if self.current_clip_duration >= self._max_clip_sec:
                            logger.info("cam-%s: max duration reached", self.cam_id)
                            self._handle_fsm_command("STOP")
                
        finally:
            cap.release()
            if self._writer:
                self._handle_fsm_command("STOP")
            self.status = "offline"
            self.person_detected = False
            self.rec_state = RecState.IDLE

    def _detect_loop(self) -> None:
        """Runs in separate thread: pulls latest frame, runs YOLO, updates FSM logic."""
        import queue
        consecutive_frames = 0
        post_roll_frames = 0
        
        while not self._stop_evt.is_set():
            try:
                frame = self._detect_q.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_area = frame.shape[0] * frame.shape[1]
            person_now, conf_max = self._detect_person(frame, frame_area)
            self.person_detected = person_now

            # FSM Status Logic (matches old flow but simplified for multi-thread)
            now_ts = time.time()
            
            if self.rec_state == RecState.IDLE:
                if person_now:
                    # Check cooldown
                    if (now_ts - self._last_clip_end_time) >= self._cooldown_sec:
                        self.rec_state = RecState.ARMED
                        consecutive_frames = 1
                        self._log("person detected, trigger armed")
            
            elif self.rec_state == RecState.ARMED:
                if person_now:
                    consecutive_frames += 1
                    if consecutive_frames >= self._min_consecutive:
                        # Success - Start recording
                        self._handle_fsm_command("START")
                        consecutive_frames = 0
                else:
                    # Lost person before confirmation
                    self.rec_state = RecState.IDLE
                    consecutive_frames = 0
                    self._log("trigger disarmed")

            elif self.rec_state == RecState.RECORDING:
                if person_now:
                    post_roll_frames = 0
                    self.detect_count += 1
                else:
                    # Person gone, enter post-roll
                    self.rec_state = RecState.POST_ROLL
                    post_roll_frames = 1
            
            elif self.rec_state == RecState.POST_ROLL:
                if person_now:
                    # Person returned!
                    self.rec_state = RecState.RECORDING
                    post_roll_frames = 0
                else:
                    post_roll_frames += 1
                    # Total post-roll frames
                    stream_fps = self.fps_actual or 25.0
                    if post_roll_frames >= int(self._post_roll_sec * stream_fps):
                        self._handle_fsm_command("STOP")
                        post_roll_frames = 0

            # Signal UI every X detections or just once
            if person_now and self.detect_count % 30 == 0:
                self._emit("detection_event", {
                    "cam_id": self.cam_id,
                    "timestamp": datetime.now().isoformat(),
                    "confidence_max": round(conf_max, 3),
                })

    def _preview_loop(self) -> None:
        """Processes snapshots for UI at limited FPS."""
        import queue
        while not self._stop_evt.is_set():
            try:
                frame = self._preview_q.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Check requested snapshot FPS
            now = time.time()
            min_interval = 1.0 / max(self._snapshot_fps, 0.001)
            if (now - self._snapshot_last_encoded_at) < min_interval:
                continue

            try:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                self._set_snapshot(buf.tobytes())
                self._snapshot_last_encoded_at = now
            except Exception:
                pass

    # ── FSM Command Actions ─────────────────────────────────────────────────
    def _handle_fsm_command(self, cmd: str) -> None:
        if cmd == "START":
            self._enforce_storage()
            w, h = self.resolution
            fps = self.fps_actual or 25.0
            self._clip_path, self._writer = self._open_writer(w, h, fps)
            self._clip_start_time = time.time()
            # Flush pre-buffer
            for f in self._pre_buffer:
                self._writer.write(f)
            self._pre_buffer.clear()
            self.rec_state = RecState.RECORDING
            logger.info("cam-%s: START recording -> %s", self.cam_id, self._clip_path.name)
            self._broadcast_status("recording")
            
        elif cmd == "STOP":
            if self._writer:
                path = self._clip_path
                self._writer.release()
                self._writer = None
                self._clip_path = None
                self._last_clip_end_time = time.time()
                self.rec_state = RecState.IDLE
                self.current_clip_duration = 0.0
                
                # Check if file is valid/long enough
                if path and path.exists():
                    duration = self._last_clip_end_time - self._clip_start_time
                    if duration < self._min_clip_sec:
                        path.unlink(missing_ok=True)
                        self._log(f"clip discarded (too short: {duration:.1f}s)")
                        logger.info("cam-%s: clip too short, discarded", self.cam_id)
                    else:
                        size_mb = path.stat().st_size / 1e6
                        self.clip_count += 1
                        logger.info("cam-%s: clip saved %s (%.1f MB)", self.cam_id, path.name, size_mb)
                        from app.api.ws_handlers import emit_clip_saved, emit_event_log
                        try:
                            data_root = Path(__file__).resolve().parents[2] / "data"
                            rel_path = path.relative_to(data_root).as_posix()
                            raw_url = f"/api/video/{rel_path}"
                        except Exception:
                            raw_url = f"/api/video/{path.name}"
                        emit_clip_saved(
                            clip_id=path.stem,
                            clip_name=path.name,
                            cam_id=self.cam_id,
                            raw_url=raw_url,
                            processed_url=None,
                            path=str(path),
                        )
                        emit_event_log("Clip queued", self.cam_id)
                        
                        if self._on_clip_ready:
                            self._on_clip_ready(path, self.cam_id)
                
                self._broadcast_status("online")

    # ── Helper methods ──────────────────────────────────────────────────────
    def _detect_person(self, frame, frame_area: int) -> tuple[bool, float]:
        if not self._model:
            return False, 0.0
        try:
            results = self._model.predict(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=self._conf_threshold,
                verbose=False,
            )
            best_conf = 0.0
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    area = float((x2 - x1) * (y2 - y1))
                    if (area / frame_area) >= self._min_box_area:
                        best_conf = max(best_conf, conf)
            return best_conf > 0, best_conf
        except Exception:
            return False, 0.0

    def _open_writer(self, w: int, h: int, fps: float) -> tuple[Path, cv2.VideoWriter]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_cam{self.cam_id}.mp4"
        path = self.output_dir / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        return path, writer

    def _push_latest(self, q: queue.Queue, item: Any) -> None:
        import queue
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _set_snapshot(self, data: bytes | None) -> None:
        with self._snap_lock:
            self._latest_jpeg = data
            self._latest_jpeg_ts = time.time() if data else 0.0

    def _enforce_storage(self) -> None:
        from app.utils.file_handler import FileHandler
        limit_gb = self._storage_limit()
        FileHandler().enforce_storage_limit(self.output_dir, limit_gb)

    def _log(self, message: str) -> None:
        from app.api.ws_handlers import emit_event_log
        emit_event_log(message, self.cam_id)

    def _broadcast_status(self, status: str, **extra):
        from app.api.ws_handlers import emit_camera_status
        is_rec = self.rec_state in (RecState.RECORDING, RecState.POST_ROLL)
        emit_camera_status(
            cam_id=self.cam_id,
            fps=round(self.fps_actual, 2),
            frame=self.detect_count,
            conf=0.9 if self.person_detected else 0.0,
            status=status if not is_rec else "RECORDING"
        )



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
        on_clip_ready: Callable[[Path, str], None] | None = None,
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
                    url=cam["rtsp_url"],
                    label=cam.get("label", f"cam{cam_id}"),
                    output_dir=output_dir,
                    model_path=model_path,
                    width=cam.get("width"),
                    height=cam.get("height"),
                    storage_limit_getter=lambda: self._storage_limit_gb,
                    socketio_emit=_emit,
                    config=self._config,
                    on_clip_ready=on_clip_ready,
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
        
        if not worker:
            logger.warning(f"No worker found for cam_id: {cam_id}")
            return None
        
        snapshot = worker.get_snapshot()
        if not snapshot:
            logger.debug(f"Worker {cam_id} returned no snapshot")
        
        return snapshot

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
