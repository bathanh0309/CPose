import time
import logging
import threading
import queue
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from app.core.engines.phase2 import run_phase2
from app.core.engines.phase3 import run_phase3
from app.core.adl_model import ADLModelWrapper
from app.utils.config_schema import AppConfig

logger = logging.getLogger("[RecognizerWorker]")

@dataclass
class Job:
    cam_id: str
    clip_path: Path
    created_at: float

class RecognizerConsumer:
    """
    Consumer: Picks clips from the queue and runs Phase 3 Sequential engines.
    """
    def __init__(self, config: AppConfig, socketio_instance):
        self.config = config
        self.socketio = socketio_instance
        self._queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Load models once
        from ultralytics import YOLO
        self.p3_model = YOLO(self.config.phase3.model) if hasattr(self.config, 'phase3') else None
        
        # Load ADL model
        try:
            adl_cfg = self.config.phase3.adl_model_cfg if hasattr(self.config.phase3, 'adl_model_cfg') else "research/common/configs/ctr_gcn_adl.yaml"
            adl_ckpt = self.config.phase3.adl_model_ckpt if hasattr(self.config.phase3, 'adl_model_ckpt') else "weights/ctr_gcn_adl.pth"
            self.adl_model = ADLModelWrapper(cfg_path=adl_cfg, ckpt_path=adl_ckpt)
        except Exception as e:
            logger.warning(f"Could not load ADL model: {e}")
            self.adl_model = None

        self._status = {
            "running": False,
            "mode": "idle",
            "current_clip": None,
            "current_cam": None,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0,
            "conf": 0,
            "adl": "unknown",
            "lamp_state": {
                "cam01": "IDLE", "cam02": "IDLE", "cam03": "IDLE", "cam04": "IDLE"
            },
            "pending_results": []
        }
        
        self._latest_original_jpeg: Optional[bytes] = None
        self._latest_processed_jpeg: Optional[bytes] = None
        self._snap_lock = threading.Lock()
        
        # In-memory storage for results waiting to be saved
        self._pending_storage = {}

    def get_status(self) -> dict:
        return self._status

    def get_snapshot(self, view: str) -> Optional[bytes]:
        with self._snap_lock:
            if view == "original":
                return self._latest_original_jpeg
            return self._latest_processed_jpeg

    def refresh_face_database(self):
        logger.info("Face database refresh requested.")
        # Logic to reload vectors if needed

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="RecognizerConsumer", daemon=True)
        self._thread.start()
        self._status["running"] = True
        logger.info("Recognizer Consumer started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._status["running"] = False

    def enqueue(self, cam_id: str, clip_path: Path):
        job = Job(cam_id=cam_id, clip_path=clip_path, created_at=time.time())
        try:
            self._queue.put_nowait(job)
            logger.info(f"Enqueued clip: {clip_path.name} from {cam_id}")
        except queue.Full:
            logger.warning(f"Queue full! Dropping clip from {cam_id}: {clip_path.name}")

    def pending_results(self) -> List[dict]:
        return list(self._pending_storage.values())

    def save_pending_result(self, clip_stem: str) -> dict:
        if clip_stem in self._pending_storage:
            item = self._pending_storage.pop(clip_stem)
            item["saved"] = True
            logger.info(f"Result for {clip_stem} saved to 'permanent' (staged).")
            return item
        return {"error": "Result not found or already saved"}

    def _worker_loop(self):
        from app.api.ws_handlers import emit_pose_progress, emit_metric_log, emit_event_log
        
        while not self._stop_event.is_set():
            try:
                job: Job = self._queue.get(timeout=1.0)
                clip_stem = job.clip_path.stem
                cam_id = job.cam_id
                
                self._status["mode"] = "processing"
                self._status["current_clip"] = clip_stem
                self._status["current_cam"] = cam_id
                self._status["lamp_state"][cam_id] = "ACTIVE"
                
                logger.info(f"Processing {clip_stem}...")
                emit_event_log(f"Processing started: {clip_stem}", cam_id)

                start_time = time.time()
                out_dir = Path("data/output_pose")
                
                if self.p3_model:
                    phase3_cfg = self.config.phase3.dict() if hasattr(self.config.phase3, 'dict') else self.config.phase3
                    
                    def _on_progress(p_data):
                        # Update status
                        self._status["current_frame"] = p_data["frame_id"]
                        self._status["total_frames"] = p_data["total_frames"]
                        self._status["adl"] = p_data["adl"]
                        self._status["conf"] = p_data["conf"]
                        
                        # Calculate FPS periodically
                        # (Simple proxy for engine FPS)
                        
                        # Emit standardized progress
                        emit_pose_progress(
                            cam_id=cam_id,
                            clip_stem=clip_stem,
                            frame_id=p_data["frame_id"],
                            total_frames=p_data["total_frames"],
                            fps=15.0, # Placeholder or calculated
                            conf=p_data["conf"],
                            adl=p_data["adl"],
                            pct=int((p_data["frame_id"] / p_data["total_frames"]) * 100) if p_data["total_frames"] > 0 else 0
                        )
                        
                        # Emit metric log
                        emit_metric_log(
                            cam=cam_id,
                            fps=15.0,
                            frame=p_data["frame_id"],
                            conf=p_data["conf"],
                            adl=p_data["adl"]
                        )
                        
                        # Update snapshots
                        with self._snap_lock:
                            if "original" in p_data:
                                _, b1 = cv2.imencode(".jpg", p_data["original"])
                                self._latest_original_jpeg = b1.tobytes()
                            if "processed" in p_data:
                                _, b2 = cv2.imencode(".jpg", p_data["processed"])
                                self._latest_processed_jpeg = b2.tobytes()

                    result = run_phase3(
                        self.p3_model, 
                        self.adl_model,
                        job.clip_path, 
                        out_dir, 
                        phase3_cfg,
                        progress_callback=_on_progress
                    )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Completed {clip_stem} in {elapsed:.2f}s")

                    # Stage result
                    self._pending_storage[clip_stem] = {
                        "clip_stem": clip_stem,
                        "cam_id": cam_id,
                        "results": result,
                        "timestamp": time.time(),
                        "saved": False
                    }
                    
                    emit_event_log(f"Finished processing: {clip_stem}", cam_id)
                    
                self._status["current_clip"] = None
                self._status["current_cam"] = None
                self._status["lamp_state"][cam_id] = "IDLE"
                self._queue.task_done()

            except queue.Empty:
                self._status["mode"] = "idle"
                continue
            except Exception as e:
                logger.error(f"Worker failure: {e}", exc_info=True)
                if 'job' in locals():
                    self._status["lamp_state"][job.cam_id] = "IDLE"
                self._queue.task_done()
