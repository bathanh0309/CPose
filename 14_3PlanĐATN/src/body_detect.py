from ultralytics import YOLO
from setting import *

class PersonDetector:
    def __init__(self, model_path = YOLO_PATH, conf_threshold= 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.person_class_id = 0

    def person_detect(self, frame):
        # input: video frame
        # output: list of coordinates and conf score of person box detected
        results = self.model(   frame, 
                                conf=self.conf_threshold, 
                                classes=[self.person_class_id], 
                                verbose=False)[0]
        detections = []
        
        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2-x1, y2-y1], score, "person"))
        return detections