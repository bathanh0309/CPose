from __future__ import annotations

import logging
import threading
import time
import math
from dataclasses import dataclass
from pathlib import Path
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

from app.core.recognizer_utils import (
    TrackState,
    TrackedDetection,
    SequentialTracker,
    PoseTemporalSmoothing,
    ADLTemporalSmoothing,
)

class RecognizerService:
    """Core logic layer for Pose & ADL recognition. Decoupled from API/SocketIO."""
    def __init__(self, socket_callback: Optional[Callable] = None, registration_callback: Optional[Callable] = None):
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._config: Dict[str, Any] = {}
        self.socket_callback = socket_callback
        self.registration_callback = registration_callback
        
        self._state: Dict[str, Any] = {
            "running": False,
            "current_clip": "",
            "active_camera": "",
            "clips_total": 0,
            "clips_done": 0,
            "progress_pct": 0,
            "lamp_state": {cam: "IDLE" for cam in MULTICAM_CAMERA_ORDER},
            "clip_queue": [],
            "error": None,
        }
        self._pending_results: Dict[str, Dict[str, Any]] = {}

    def start(self, clips: List[Path], output_dir: Path, model_path: Path, config_path: Path, save_overlay: bool = True):
        if self.is_running(): return
        
        ordered_clips = sort_multicam_clips(clips)
        self._stop_evt.clear()
        
        with self._lock:
            self._state.update({
                "running": True,
                "clips_total": len(ordered_clips),
                "clips_done": 0,
                "progress_pct": 0,
                "lamp_state": {cam: "IDLE" for cam in MULTICAM_CAMERA_ORDER},
                "clip_queue": self._build_queue_info(ordered_clips),
                "error": None
            })

        self._thread = threading.Thread(
            target=self._run,
            args=(ordered_clips, output_dir, model_path, save_overlay),
            daemon=True,
            name="core-recognizer"
        )
        self._thread.start()

    def stop(self):
        self._stop_evt.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {k: v for k, v in self._state.items() if not k.startswith("snapshot_")}

    def get_snapshot(self, view: str) -> Optional[bytes]:
        with self._lock:
            return self._state.get(f"snapshot_{view}")

    def pending_results(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._pending_results)

    def save_pending_result(self, clip_stem: str) -> Dict[str, Any]:
        with self._lock:
            if clip_stem not in self._pending_results:
                return {"error": "No pending result"}
            result = self._pending_results[clip_stem]
            result["saved"] = True
            return {"ok": True, "clip_stem": clip_stem}

    def _update_state(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)

    def _build_queue_info(self, clips: List[Path]):
        return [{"clip_stem": c.stem, "cam_id": extract_multicam_camera_id(c) or "unknown"} for c in clips]

    def _emit(self, event: str, data: Any):
        if self.socket_callback:
            self.socket_callback(event, data)

    def _run(self, clips: List[Path], output_dir: Path, model_path: Path, save_overlay: bool):
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            adl_model = None # Loaded if needed
        except Exception as e:
            self._update_state(running=False, error=str(e))
            self._emit("error", {"message": f"Initialization failed: {e}"})
            return

        clips_done = 0
        processed_clips = []
        
        for clip in clips:
            if self._stop_evt.is_set(): break
            
            cam_id = extract_multicam_camera_id(clip) or ""
            lamp_state = dict(self.status()["lamp_state"])
            lamp_state[cam_id] = "ACTIVE"
            self._update_state(current_clip=clip.name, active_camera=cam_id, lamp_state=lamp_state)
            self._emit("pose_lamp_state", self.status())

            # Face registration hook
            if cam_id == "cam01" and self.registration_callback:
                self.registration_callback(clip.stem, cam_id)

            try:
                # Actual clip processing logic
                from app.core.engines.phase3 import run_phase3
                run_phase3(model, adl_model, clip, output_dir, _PHASE3_CFG, save_overlay)
                processed_clips.append(clip)
                
                # Global ID Merger hook
                merger = CrossCameraIDMerger(output_dir, processed_clips)
                merger.merge()
                
                # Stage result
                with self._lock:
                    self._pending_results[clip.stem] = {
                        "clip_stem": clip.stem,
                        "temp_dir": str(output_dir / clip.stem),
                        "saved": False
                    }
            except Exception as e:
                logger.exception(f"Error processing {clip.stem}: {e}")
                lamp_state[cam_id] = "ALERT"
                self._update_state(lamp_state=lamp_state)
            
            clips_done += 1
            lamp_state = dict(self.status()["lamp_state"])
            if lamp_state.get(cam_id) != "ALERT":
                lamp_state[cam_id] = "DONE"
                
            self._update_state(
                clips_done=clips_done,
                progress_pct=int((clips_done / len(clips)) * 100),
                lamp_state=lamp_state
            )
            self._emit("pose_lamp_state", self.status())

        self._update_state(running=False)
        self._emit("pose_complete", self.status())
