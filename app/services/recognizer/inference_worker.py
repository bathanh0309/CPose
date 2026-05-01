import time
import logging
import threading
from queue import Queue, Empty
from typing import Optional

from app.services.recognizer.multicam_online_system import MultiCamOnlineSystem

logger = logging.getLogger("[InferenceWorker]")

class InferenceWorker:
    """
    Consumer: Reads frames from the queue, passes to AI Core for processing.
    """
    def __init__(self, frame_queue: Queue, publisher_queue: Queue, ai_system: MultiCamOnlineSystem):
        self.frame_queue = frame_queue
        self.publisher_queue = publisher_queue
        self.ai_system = ai_system
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="InferenceWorker",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        logger.info("Inference worker started.")
        while not self._stop_event.is_set():
            try:
                # Wait for frames with timeout to periodically check stop_event
                cam_id, frame, timestamp = self.frame_queue.get(timeout=0.1)
                
                # Push frame to AI Core System
                # Note: This is an intensive operation that blocks this thread,
                # but keeps the RTSP queue free in the Ingestor threads!
                self.ai_system.process_frame(cam_id, frame, timestamp)
                
                # Fetch updated tracks
                ts, tracks = self.ai_system.get_current_tracks()
                
                # Send to publisher
                self.publisher_queue.put((ts, tracks))
                
                self.frame_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Inference worker error: {e}", exc_info=True)

        logger.info("Inference worker stopped.")
