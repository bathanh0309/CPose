from typing import List
import numpy as np
from ..model_provider import get_model

class YOLOUltraDetector(BaseDetector):
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = get_model(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            classes=[0],  # Class 0 is usually 'person' in COCO
            verbose=False
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf)
                cls = int(box.cls)
                
                detections.append(Detection(
                    bbox=b.tolist(),
                    confidence=conf,
                    class_id=cls,
                    label=self.model.names[cls]
                ))
        return detections
