from typing import List
from .generic_detector import Detection, BaseDetector

class PersonDetector(BaseDetector):
    """
    Wrapper for YOLOv8/v11/RTMDet person detection.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # TODO: Load YOLO model using ultralytics or similar
        self.model = None

    def detect(self, image) -> List[Detection]:
        """
        Input: image (numpy array, BGR)
        Output: List of Detection objects (bbox, score, label='person')
        """
        # TODO: Run inference
        # result = self.model.predict(image, classes=[0], conf=self.conf_threshold)
        detections = []
        return detections

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            model_path=cfg.get("model_path", "models/product/yolo11n.pt"),
            conf_threshold=cfg.get("conf", 0.5)
        )
