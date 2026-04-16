import logging
from pathlib import Path
from typing import Dict, Any, List

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
