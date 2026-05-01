from typing import List
import numpy as np
from ..model_provider import get_model

class YOLOPoseEstimator(BasePoseEstimator):
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = get_model(model_path)
        self.conf_threshold = conf_threshold

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            task='pose',
            verbose=False
        )
        
        pose_results = []
        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints
            if keypoints is None:
                continue
                
            for i in range(len(boxes)):
                box = boxes[i]
                kps = keypoints.data[i].cpu().numpy() # [17, 3] or [17, 2]
                
                b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                pose_results.append(PoseResult(
                    keypoints=kps,
                    bbox=b.tolist(),
                    confidence=conf
                ))
        return pose_results
