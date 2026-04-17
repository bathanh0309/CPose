import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .worker import RecognizerConsumer
from app.utils.config_schema import AppConfig

logger = logging.getLogger("[RecognizerService]")

class RecognizerService:
    """
    Orchestrator for the Clip-based (Sequential) AI Pipeline.
    Manages the job queue and ensures consumer workers are running.
    """
    def __init__(self, config: AppConfig, socketio_instance):
        self.config = config
        self.consumer = RecognizerConsumer(config, socketio_instance)
        
    def start(self):
        """Starts the background worker threads."""
        self.consumer.start()
        
    def stop(self):
        """Stops the background worker threads."""
        self.consumer.stop()
        
    def enqueue_clip(self, cam_id: str, clip_path: Path):
        """Public API for Phase 1 to enqueue a saved clip for analysis."""
        self.consumer.enqueue(cam_id, clip_path)
        
    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the consumer and queue."""
        return self.consumer.status

    def get_snapshot(self, view: str) -> Optional[bytes]:
        """Return latest snapshot (original or processed) from the consumer."""
        return self.consumer.get_snapshot(view)

    def pending_results(self) -> Dict[str, Any]:
        return {} # Placeholder or implement if needed

    def save_pending_result(self, clip_stem: str) -> Dict[str, Any]:
        return {"ok": True} # Placeholder

    def refresh_face_database(self):
        """Reload embeddings in the AI pipeline."""
        if hasattr(self.consumer, "refresh_face_database"):
            self.consumer.refresh_face_database()
