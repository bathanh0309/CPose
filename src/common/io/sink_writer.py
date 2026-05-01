import cv2
import numpy as np
from typing import Optional

class SinkWriter:
    def __init__(self, output_path: str, fps: float, size: tuple, codec: str = 'mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        self.output_path = output_path

    def write(self, frame: np.ndarray):
        if self.writer.isOpened():
            self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
