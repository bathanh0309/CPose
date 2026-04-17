import time
import logging
import threading
import queue
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
    Consumer: Picks clips from the queue and runs Phase 2/3 Sequential engines.
    """
    def __init__(self, config: AppConfig, socketio_instance):
        self.config = config
        self.socketio = socketio_instance
        self._queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Load models once
        from ultralytics import YOLO
        self.p2_model = YOLO(self.config.phase2.model) if hasattr(self.config, 'phase2') else None
        self.p3_model = YOLO(self.config.phase3.model) if hasattr(self.config, 'phase3') else None
        
        # Load ADL model
        try:
            adl_cfg = self.config.phase3.adl_model_cfg if hasattr(self.config.phase3, 'adl_model_cfg') else "configs/ctr_gcn_adl.yaml"
            adl_ckpt = self.config.phase3.adl_model_ckpt if hasattr(self.config.phase3, 'adl_model_ckpt') else "weights/ctr_gcn_adl.pth"
            self.adl_model = ADLModelWrapper(cfg_path=adl_cfg, ckpt_path=adl_ckpt)
        except Exception as e:
            logger.warning(f"Could not load ADL model: {e}")
            self.adl_model = None

        self.status = {
            "is_running": False,
            "jobs_pending": 0,
            "current_job": None,
            "completed_total": 0
        }
        self._latest_snap: Optional[bytes] = None
        self._snap_lock = threading.Lock()

    def get_snapshot(self, view: str) -> Optional[bytes]:
        with self._snap_lock:
            return self._latest_snap

    def refresh_face_database(self):
        logger.info("Face database refresh requested.")
        # Logic to reload vectors if needed

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="RecognizerConsumer", daemon=True)
        self._thread.start()
        self.status["is_running"] = True
        logger.info("Recognizer Consumer started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.status["is_running"] = False

    def enqueue(self, cam_id: str, clip_path: Path):
        job = Job(cam_id=cam_id, clip_path=clip_path, created_at=time.time())
        try:
            self._queue.put_nowait(job)
            self.status["jobs_pending"] = self._queue.qsize()
            logger.info(f"Enqueued clip: {clip_path.name} from {cam_id}")
        except queue.Full:
            logger.warning(f"Queue full! Dropping clip from {cam_id}: {clip_path.name}")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job: Job = self._queue.get(timeout=1.0)
                self.status["jobs_pending"] = self._queue.qsize()
                self.status["current_job"] = job.clip_path.name
                
                logger.info(f"Starting processing for {job.clip_path.name}...")
                
                # Update UI state to ACTIVE
                self.socketio.emit("pose_lamp_state", {
                    "active_camera": job.cam_id,
                    "current_clip": job.clip_path.name,
                    "lamp_state": {job.cam_id: "ACTIVE"}
                })

                start_time = time.time()
                
                # 1. Phase 2 (If enabled or needed)
                # Output dir based on config
                out_dir = Path("data/output_pose")
                
                # 2. Phase 3 (Pose & ADL)
                if self.p3_model:
                    # Pass config as dict for the engine
                    phase3_cfg = self.config.phase3.dict() if hasattr(self.config.phase3, 'dict') else self.config.phase3
                    
                    def _on_progress(p_data):
                        # p_data: {frame_id, total_frames, adl, conf, frame}
                        pct = int((p_data["frame_id"] / p_data["total_frames"]) * 100) if p_data["total_frames"] > 0 else 0
                        
                        # Emit progress
                        self.socketio.emit("pose_progress", {
                            "clip_id": job.clip_path.name,
                            "cam_id": job.cam_id,
                            "frame_id": p_data["frame_id"],
                            "total_frames": p_data["total_frames"],
                            "adl": p_data["adl"],
                            "conf": round(p_data["conf"], 2),
                            "pct": pct
                        })
                        
                        # Update snapshot
                        if "frame" in p_data:
                            _, buf = cv2.imencode(".jpg", p_data["frame"])
                            with self._snap_lock:
                                self._latest_snap = buf.tobytes()

                    result = run_phase3(
                        self.p3_model, 
                        self.adl_model,
                        job.clip_path, 
                        out_dir, 
                        phase3_cfg,
                        progress_callback=_on_progress
                    )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Completed {job.clip_path.name} in {elapsed:.2f}s. KPs: {result['keypoints_written']}")

                    # Emit completion
                    self.socketio.emit("pose_complete", {
                        "cam_id": job.cam_id,
                        "clip": job.clip_path.name,
                        "results": result,
                        "elapsed": elapsed
                    })
                    
                    # Reset lamp to IDLE or DONE
                    self.socketio.emit("pose_lamp_state", {
                        "active_camera": "",
                        "lamp_state": {job.cam_id: "IDLE"}
                    })

                self.status["completed_total"] += 1
                self.status["current_job"] = None
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker iteration failed: {e}", exc_info=True)
                self._queue.task_done()
