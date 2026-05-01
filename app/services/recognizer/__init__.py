import logging
from queue import Queue
from typing import Dict, List, Optional

from app.services.recognizer.multicam_online_system import MultiCamOnlineSystem

from .stream_ingestor import StreamIngestor
from .inference_worker import InferenceWorker
from .event_publisher import EventPublisher

logger = logging.getLogger("[RecognizerOrchestrator]")

class RecognizerService:
    """
    Main entry point for real-time AI recognition.
    Orchestrates the Ingestor (Producer), InferenceWorker (Consumer),
    and EventPublisher (Delivery pipeline) under a unified interface.
    """
    def __init__(self, ai_system: MultiCamOnlineSystem):
        self.ai_system = ai_system
        
        # Internal Queues for passing states
        # Maxsize controls backpressure to prevent Memory errors on huge lag
        self.frame_queue = Queue(maxsize=30) 
        self.publisher_queue = Queue(maxsize=30)
        
        # Components
        self.ingestors: Dict[str, StreamIngestor] = {}
        self.inference_worker = InferenceWorker(self.frame_queue, self.publisher_queue, self.ai_system)
        self.publisher = EventPublisher(self.publisher_queue)
        
        self.is_running = False

    def start_all(self, camera_streams: List[dict]):
        """
        Starts the publisher, worker, and all registered streams.
        camera_streams: list of {"id": "cam01", "url": "rtsp://..."}
        """
        if self.is_running:
            return
            
        logger.info("Starting RecognizerService pipeline...")
        self.publisher.start()
        self.inference_worker.start()
        
        for cam in camera_streams:
            cam_id = cam.get("id")
            source = cam.get("url")
            if cam_id and source:
                self.add_camera(cam_id, source)
                
        self.is_running = True

    def stop_all(self):
        """Stops all running pipelines gracefully."""
        logger.info("Stopping RecognizerService pipeline...")
        for cam_id, ingestor in self.ingestors.items():
            ingestor.stop()
        self.ingestors.clear()
        
        self.inference_worker.stop()
        self.publisher.stop()
        
        # Drain queues
        self.frame_queue.queue.clear()
        self.publisher_queue.queue.clear()
        self.is_running = False

    def add_camera(self, cam_id: str, source: str):
        if cam_id in self.ingestors:
            logger.warning(f"Camera {cam_id} is already running.")
            return
            
        ingestor = StreamIngestor(cam_id, source, self.frame_queue)
        self.ingestors[cam_id] = ingestor
        ingestor.start()

    def remove_camera(self, cam_id: str):
        if cam_id not in self.ingestors:
            return
            
        logger.info(f"Removing camera stream: {cam_id}")
        ingestor = self.ingestors.pop(cam_id)
        ingestor.stop()

# Example hook for global app initialization
# _service = None
# def init_recognizer(app_config) -> RecognizerService:
#     global _service
#     ai_core = MultiCamOnlineSystem(app_config)
#     _service = RecognizerService(ai_core)
#     return _service
