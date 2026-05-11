import cv2
import time
from threading import Thread, Lock
from .sync import CameraSync, get_camera_fps

class CameraWorker:
    def __init__(self, name, rtsp_url, pipeline, target_fps = 15):
        self.name = name
        self.cap = cv2.VideoCapture(rtsp_url)
        self.pipeline = pipeline
        
        self.sync = CameraSync(target_fps)
        
        self.lock = Lock()        
        self.running = True
        
        self.frame = None
        self.results = []
        self.draw_data = []
        
        # detect fps
        self.original_fps = get_camera_fps(self.cap)
        print(f"[{name} Original FPS: {self.original_fps}]")

    def start(self):
        Thread(target=self.run, daemon=True).start()
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            timestamp = time.time() # global timestamp
            
            results = self.pipeline.process(frame, timestamp)
            
            with self.lock:
                self.frame = frame.copy()
                self.results = results
                
            self.sync.wait()
            
    def get_results(self):
        with self.lock:
            return self.frame, self.results
