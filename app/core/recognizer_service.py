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
    def __init__(
        self,
        socket_callback: Optional[Callable] = None,
        registration_callback: Optional[Callable] = None,
        pose_model_path: Path | str | None = None,
    ):
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._config: Dict[str, Any] = {}
        self.socket_callback = socket_callback
        self.registration_callback = registration_callback
        self._pose_model_path = self._resolve_pose_model_path(pose_model_path)
        
        # Producer-Consumer queue
        self._queue = queue.Queue(maxsize=100)
        self._last_metric_emit_at = 0.0
        self._metric_emit_interval = 0.30
        
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
        self._clear_snapshots()
        self._last_metric_emit_at = 0.0
        self._thread = threading.Thread(target=self._worker_loop, name="CoreRecognizerWorker", daemon=True)
        self._thread.start()
        self._update_state(running=True)
        logger.info("Core Recognizer Worker started.")

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._clear_snapshots()
        self._update_state(
            running=False,
            mode="idle",
            current_clip=None,
            active_camera=None,
            current_frame=0,
            total_frames=0,
            fps=0,
            conf=0,
            adl="unknown",
        )

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
        model_path = self._pose_model_path
        model = YOLO(str(model_path))
        adl_model = None # ADLModelWrapper(..) can be loaded here

        while not self._stop_evt.is_set():
            try:
                job: Job = self._queue.get(timeout=1.0)
                clip_stem = job.clip_path.stem
                cam_id = job.cam_id
                self._last_metric_emit_at = 0.0
                
                self._update_state(
                    mode="processing",
                    current_clip=clip_stem,
                    active_camera=cam_id,
                    current_frame=0,
                    total_frames=0,
                    fps=0,
                    conf=0,
                    adl="unknown",
                )
                lamp_state = dict(self.status()["lamp_state"])
                lamp_state[cam_id] = "ACTIVE"
                self._update_state(lamp_state=lamp_state)
                
                self._emit("pose_lamp_state", {cam_id: "ACTIVE"})
                try:
                    from app.api.ws_handlers import emit_workspace_state
                    emit_workspace_state(
                        mode="multicam_folder",
                        running=True,
                        current_clip=clip_stem,
                        current_cam=cam_id,
                        output_dir="output_pose",
                        queued=self._queue.qsize(),
                    )
                except Exception:
                    logger.debug("workspace_state emit failed at clip start", exc_info=True)

                from app.api.ws_handlers import emit_event_log
                emit_event_log("Clip processing", cam_id)

                # Prime the snapshot endpoints immediately so the UI has a frame
                # before phase3 emits its first progress callback.
                self._prime_initial_snapshots(job.clip_path, cam_id, clip_stem)

                # Keep the registration hook non-blocking so one clip cannot stall the queue.
                if cam_id == "cam01" and self.registration_callback:
                    threading.Thread(
                        target=self.registration_callback,
                        args=(clip_stem, cam_id),
                        name=f"RegistrationRequest-{clip_stem}",
                        daemon=True,
                    ).start()

                def _progress_cb(p_data):
                    if self._stop_evt.is_set():
                        return
                    # p_data: {frame_id, total_frames, adl, conf, original, processed}
                    now_ts = time.time()
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
                    if (now_ts - self._last_metric_emit_at) >= self._metric_emit_interval or p_data["frame_id"] >= max(p_data["total_frames"] - 1, 0):
                        emit_metric_log(cam=cam_id, fps=15.0, frame=p_data["frame_id"], conf=p_data["conf"], adl=p_data["adl"])
                        self._last_metric_emit_at = now_ts

                from app.core.engines.phase3 import run_phase3
                out_dir = Path("data/output_pose")
                result = run_phase3(
                    model,
                    adl_model,
                    job.clip_path,
                    out_dir,
                    _PHASE3_CFG,
                    progress_callback=_progress_cb,
                    stop_event=self._stop_evt,
                )

                if self._stop_evt.is_set():
                    logger.info("Stop requested while processing %s; skipping result emit.", clip_stem)
                    self._queue.task_done()
                    continue

                # Stage result
                with self._lock:
                    self._pending_results[clip_stem] = {
                        "clip_stem": clip_stem,
                        "cam_id": cam_id,
                        "results": result,
                        "timestamp": time.time(),
                        "saved": False
                    }

                processed_url = None
                processed_path = result.get("processed_video_path") if isinstance(result, dict) else None
                if processed_path:
                    try:
                        processed_path = Path(processed_path)
                        data_root = Path(__file__).resolve().parents[2] / "data"
                        rel_processed = processed_path.relative_to(data_root).as_posix()
                        processed_url = f"/api/video/{rel_processed}"
                    except Exception:
                        processed_url = None

                raw_url = None
                try:
                    data_root = Path(__file__).resolve().parents[2] / "data"
                    rel_raw = job.clip_path.relative_to(data_root).as_posix()
                    raw_url = f"/api/video/{rel_raw}"
                except Exception:
                    raw_url = f"/api/video/{job.clip_path.name}"

                from app.api.ws_handlers import emit_clip_saved
                emit_clip_saved(
                    clip_id=clip_stem,
                    clip_name=job.clip_path.name,
                    cam_id=cam_id,
                    raw_url=raw_url,
                    processed_url=processed_url,
                )
                emit_event_log("Clip done", cam_id)
                
                lamp_state[cam_id] = "IDLE"
                self._update_state(mode="idle", current_clip=None, active_camera=None, lamp_state=lamp_state)
                self._emit("pose_complete", {"clip": clip_stem, "cam": cam_id})
                try:
                    from app.api.ws_handlers import emit_workspace_state
                    emit_workspace_state(
                        mode="multicam_folder",
                        running=True,
                        current_clip=clip_stem,
                        current_cam=cam_id,
                        output_dir="output_pose",
                        queued=self._queue.qsize(),
                    )
                except Exception:
                    logger.debug("workspace_state emit failed at clip finish", exc_info=True)
                self._queue.task_done()
                if self._queue.empty():
                    self._finalize_multicam_if_idle()

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker iteration failure: {e}")
                self._queue.task_done()
                if self._queue.empty():
                    self._finalize_multicam_if_idle()

    def _resolve_pose_model_path(self, pose_model_path: Path | str | None) -> Path:
        candidate = Path(pose_model_path) if pose_model_path else Path("models/product/yolov8n-pose.pt")
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parents[2] / candidate

    def _clear_snapshots(self) -> None:
        with self._lock:
            self._state["snapshot_original"] = None
            self._state["snapshot_processed"] = None

    def _prime_initial_snapshots(self, clip_path: Path, cam_id: str, clip_stem: str) -> None:
        cap = cv2.VideoCapture(str(clip_path))
        try:
            if not cap.isOpened():
                logger.debug("Unable to prime snapshots for %s (%s): clip cannot be opened", clip_stem, cam_id)
                return

            ok, frame = cap.read()
            if not ok or frame is None:
                logger.debug("Unable to prime snapshots for %s (%s): first frame unavailable", clip_stem, cam_id)
                return

            jpeg = self._encode_jpeg(frame)
            if not jpeg:
                return

            self._update_state(
                current_frame=0,
                total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0,
                snapshot_original=jpeg,
                snapshot_processed=jpeg,
            )
        finally:
            cap.release()

    def _encode_jpeg(self, frame) -> Optional[bytes]:
        if frame is None: return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def _finalize_multicam_if_idle(self) -> None:
        lamp_state = dict(self.status().get("lamp_state") or {})
        for key in list(lamp_state.keys()):
            lamp_state[key] = "IDLE"

        workspace_mode = "multicam_folder"
        output_dir = "output_pose"
        staged_clips: list[str] = []
        staged_camera_map: list[dict] = []

        try:
            from app.api import routes as api_routes

            with api_routes._WORKSPACE_LOCK:
                workspace_mode = api_routes._WORKSPACE.mode or workspace_mode
                output_dir = api_routes._WORKSPACE.output_dir or output_dir
                staged_clips = [str(path) for path in api_routes._WORKSPACE.staged_clips]
                staged_camera_map = list(api_routes._WORKSPACE.staged_camera_map)
                if not str(workspace_mode).startswith("multicam"):
                    return
                api_routes._WORKSPACE.running = False
                api_routes._WORKSPACE.current_clip = None
                api_routes._WORKSPACE.current_cam = None
        except Exception:
            logger.debug("Workspace idle sync failed.", exc_info=True)
            return

        self._clear_snapshots()
        self._update_state(
            running=False,
            mode="idle",
            current_clip=None,
            active_camera=None,
            current_frame=0,
            total_frames=0,
            fps=0,
            conf=0,
            adl="unknown",
            lamp_state=lamp_state,
        )

        try:
            from app.api.ws_handlers import emit_event_log, emit_workspace_state

            emit_event_log("MC finished", "0")
            emit_workspace_state(
                mode=workspace_mode,
                running=False,
                current_clip=None,
                current_cam=None,
                output_dir=output_dir,
                queued=len(staged_clips),
                staged_clips=staged_clips,
                staged_camera_map=staged_camera_map,
            )
        except Exception:
            logger.debug("workspace_state emit failed at multicam finish", exc_info=True)
