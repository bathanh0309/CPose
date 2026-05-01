import cv2
import time
import logging
import threading
from queue import Queue, Full
from typing import Optional

logger = logging.getLogger("[Ingestor]")

class StreamIngestor:
    """
    Producer: Reads frames from RTSP or Video File and pushes them into a Queue.
    Operates on its own thread to ensure cv2.VideoCapture doesn't block AI ML.
    """
    def __init__(self, cam_id: str, source: str, frame_queue: Queue, target_fps: int = 15):
        self.cam_id = cam_id
        self.source = source
        self.frame_queue = frame_queue
        self.target_fps = target_fps
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"Ingestor-{self.cam_id}",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        # Apply optimal configs for low latency RTSP if applicable
        if str(self.source).startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.source}")
            return

        frame_time = 1.0 / self.target_fps
        logger.info(f"Started ingestion for {self.cam_id} from {self.source}")

        while not self._stop_event.is_set():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {self.cam_id}. Reconnecting...")
                # Typically, you'd add reconnection logic here
                time.sleep(1)
                continue

            # Push frame to queue without blocking. Drop frame if queue is full.
            try:
                self.frame_queue.put_nowait((self.cam_id, frame, loop_start))
            except Full:
                # Optionally warn about dropped frames
                pass

            # Sleep to maintain target FPS
            process_time = time.time() - loop_start
            sleep_time = max(0, frame_time - process_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        logger.info(f"Ingestion stopped for {self.cam_id}")
