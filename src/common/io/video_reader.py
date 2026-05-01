import cv2
from typing import Generator, Tuple, Optional
import numpy as np
import time

def open_video(source: str) -> Generator[Tuple[np.ndarray, float, int], None, None]:
    """
    Generator yielding (frame, timestamp, frame_idx)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate approximate timestamp if not available
        ts = frame_idx / fps
        yield frame, ts, frame_idx
        frame_idx += 1
        
    cap.release()
