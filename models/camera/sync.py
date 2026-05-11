import time
import cv2

class CameraSync:
    def __init__(self, target_fps=15):
        self.frame_target = target_fps
        self.frame_interval  = 1.0 / target_fps
        self.last_time = time.time()
        
    def wait(self):
        now = time.time()
        elapsed = now - self.last_time
        
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
            
        self.last_time = time.time()
        
def get_camera_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0 or fps > 60:
        fps = 25 # fall back
        
    return fps