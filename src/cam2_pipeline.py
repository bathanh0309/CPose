from models.tracking.tracking import Tracker
from models.human_detect.human_detect import PersonDetector

class Cam2Pipeline:
    def __init__(self):
        self.tracker = Tracker()
        self.person_detection = PersonDetector()
        
    def process(self, frame, timestamp):
        detections = self.person_detection.person_detect(frame)
                
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        results = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            foot_x = (x1+x2)//2
            foot_y = y2
            
            results.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                
                "label": f"ID {track_id}",
                "color": (0,255,255),
                
                "footpoint": (foot_x, foot_y),
                
                "timestamp": timestamp,
            })
                                    
        return results