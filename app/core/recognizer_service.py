from __future__ import annotations
import logging
import threading
import time
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, deque

import cv2
import numpy as np

from app.utils.file_handler import (
    MULTICAM_CAMERA_ORDER,
    extract_multicam_camera_id,
    multicam_sort_key,
    sort_multicam_clips,
)
from app.utils.runtime_config import get_runtime_section
from app.utils.pose_utils import draw_skeleton, rule_based_adl
from app.core.adl_model import ADLModelWrapper
from app.core.cross_camera import CrossCameraIDMerger

logger = logging.getLogger("[CoreRecognizer]")

_PHASE3_CFG = get_runtime_section("phase3")
CONF_THRESHOLD_P3 = float(_PHASE3_CFG.get("conf_threshold", 0.45))
KP_CONF_MIN = float(_PHASE3_CFG.get("keypoint_conf_min", 0.30))
WINDOW_SIZE = int(_PHASE3_CFG.get("window_size", 30))
PROGRESS_EVERY = int(_PHASE3_CFG.get("progress_every", 10))

@dataclass
class Job:
    cam_id: str
    clip_path: Path
    created_at: float

class RecognizerService:
    """Core logic layer for Pose & ADL recognition. Decoupled from API/SocketIO."""
    def __init__(self, socket_callback: Optional[Callable] = None, registration_callback: Optional[Callable] = None):
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._config: Dict[str, Any] = {}
        self.socket_callback = socket_callback
        self.registration_callback = registration_callback
        
        # Producer-Consumer queue
        self._queue = queue.Queue(maxsize=100)
        
        self._state: Dict[str, Any] = {
            "running": False,
            "mode": "idle",
            "current_clip": None,
            "active_camera": None,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0,
            "conf": 0,
            "adl": "unknown",
            "lamp_state": {cam: "IDLE" for cam in MULTICAM_CAMERA_ORDER},
            "pending_results": [],
            "error": None,
        }
        self._pending_results: Dict[str, Dict[str, Any]] = {}

    def start_worker(self):
        """Starts the background consumer worker thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="CoreRecognizerWorker", daemon=True)
        self._thread.start()
        self._update_state(running=True)
        logger.info("Core Recognizer Worker started.")

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._update_state(running=False)

    def enqueue_clip(self, cam_id: str, clip_path: Path):
        """Public API to enqueue a saved clip for analysis."""
        job = Job(cam_id=cam_id, clip_path=clip_path, created_at=time.time())
        try:
            self._queue.put_nowait(job)
            logger.info(f"Enqueued: {clip_path.name} from {cam_id}")
        except queue.Full:
            logger.warning(f"Queue full! Dropped {clip_path.name}")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            # Flatten pending results for the UI
            res = {k: v for k, v in self._state.items() if not k.startswith("snapshot_")}
            res["pending_results"] = list(self._pending_results.values())
            return res

    def get_snapshot(self, view: str) -> Optional[bytes]:
        with self._lock:
            return self._state.get(f"snapshot_{view}")

    def pending_results(self) -> List[dict]:
        with self._lock:
            return list(self._pending_results.values())

    def save_pending_result(self, clip_stem: str) -> Dict[str, Any]:
        with self._lock:
            if clip_stem in self._pending_results:
                item = self._pending_results.pop(clip_stem)
                item["saved"] = True
                return {"ok": True, "clip_stem": clip_stem}
            return {"error": "Result not found"}

    def refresh_face_database(self):
        logger.info("Face database refresh requested.")
        # Re-load models if needed

    def _update_state(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)

    def _emit(self, event: str, data: Any):
        if self.socket_callback:
            self.socket_callback(event, data)

    def _worker_loop(self):
        from ultralytics import YOLO
        # Pre-load model
        model_path = Path("models/product/yolov8n-pose.pt") # Default
        model = YOLO(str(model_path))
        adl_model = None # ADLModelWrapper(..) can be loaded here

        while not self._stop_evt.is_set():
            try:
                job: Job = self._queue.get(timeout=1.0)
                clip_stem = job.clip_path.stem
                cam_id = job.cam_id
                
                self._update_state(mode="processing", current_clip=clip_stem, active_camera=cam_id)
                lamp_state = dict(self.status()["lamp_state"])
                lamp_state[cam_id] = "ACTIVE"
                self._update_state(lamp_state=lamp_state)
                
                self._emit("pose_lamp_state", {cam_id: "ACTIVE"})

                # Registration hook for specific cams if needed
                if cam_id == "cam01" and self.registration_callback:
                    self.registration_callback(clip_stem, cam_id)

                def _progress_cb(p_data):
                    # p_data: {frame_id, total_frames, adl, conf, original, processed}
                    self._update_state(
                        current_frame=p_data["frame_id"],
                        total_frames=p_data["total_frames"],
                        adl=p_data["adl"],
                        conf=p_data["conf"],
                        snapshot_original=self._encode_jpeg(p_data["original"]),
                        snapshot_processed=self._encode_jpeg(p_data["processed"])
                    )
                    
                    # Also use emitters if available in some scope
                    from app.api.ws_handlers import emit_pose_progress, emit_metric_log
                    emit_pose_progress(
                        cam_id=cam_id,
                        clip_stem=clip_stem,
                        frame_id=p_data["frame_id"],
                        total_frames=p_data["total_frames"],
                        fps=15.0,
                        conf=p_data["conf"],
                        adl=p_data["adl"],
                        pct=int(p_data["frame_id"] * 100 / max(p_data["total_frames"],1))
                    )
                    emit_metric_log(cam=cam_id, fps=15.0, frame=p_data["frame_id"], conf=p_data["conf"], adl=p_data["adl"])

                from app.core.engines.phase3 import run_phase3
                out_dir = Path("data/output_pose")
                result = run_phase3(model, adl_model, job.clip_path, out_dir, _PHASE3_CFG, progress_callback=_progress_cb)

                # Stage result
                with self._lock:
                    self._pending_results[clip_stem] = {
                        "clip_stem": clip_stem,
                        "cam_id": cam_id,
                        "results": result,
                        "timestamp": time.time(),
                        "saved": False
                    }
                
                lamp_state[cam_id] = "IDLE"
                self._update_state(mode="idle", current_clip=None, active_camera=None, lamp_state=lamp_state)
                self._emit("pose_complete", {"clip": clip_stem, "cam": cam_id})
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker iteration failure: {e}")
                self._queue.task_done()

    def _encode_jpeg(self, frame) -> Optional[bytes]:
        if frame is None: return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
