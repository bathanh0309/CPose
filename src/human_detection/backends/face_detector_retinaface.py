from typing import List
from .generic_detector import Detection, BaseDetector

class FaceDetectorRetinaFace(BaseDetector):
    """
    Wrapper for RetinaFace/SCRFD face detection.
    Used for face identification/registration.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # TODO: Load RetinaFace/SCRFD model
        self.model = None

    def detect(self, image) -> List[Detection]:
        """
        Detect faces in image.
        Returns bboxes and landmarks (optional in Detection class but needed for alignment).
        """
        detections = []
        return detections
