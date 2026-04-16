import time
import logging
import threading
from queue import Queue, Empty
from typing import Optional

from app.api.ws_handlers import push_camera_update

logger = logging.getLogger("[EventPublisher]")

class EventPublisher:
    """
    Subscribes to finalized track states from inference worker
    and pushes them over WebSocket efficiently.
    """
    def __init__(self, publisher_queue: Queue):
        self.publisher_queue = publisher_queue
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="EventPublisher",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        logger.info("Event Publisher started.")
        while not self._stop_event.is_set():
            try:
                # Retrieve inferred states
                ts, tracks = self.publisher_queue.get(timeout=0.1)
                
                # Push to all connected clients.
                # If there are many updates in rapid succession, you might decimate 
                # or batch them here so you don't flood SocketIO.
                push_camera_update(tracks, ts)
                
                self.publisher_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Publisher error: {e}", exc_info=True)

        logger.info("Event Publisher stopped.")
