import cv2
import time

class VideoSource:
    def __init__ (self, source, name="cam"):
        self.source = source
        self.name = name
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # fall if video dont have fps
        if self.fps <= 0 or self.fps > 60:
            self.fps = 15
            
        print(f"[{name}] Source: {source}")
        print(f"[{name}] FPS: {self.fps}")
        
        self.frame_interval = 1.0 / self.fps
        self.last_time = time.time()
        
    def read(self):
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        timestamp = time.time()
        
        return frame, timestamp
    
    def release(self):
        self.cap.release()