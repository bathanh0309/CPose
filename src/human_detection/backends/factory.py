from .base import BaseDetector
from .yolo_ultra import YOLOUltraDetector

def build_detector(name: str, cfg: dict) -> BaseDetector:
    """
    Factory to create detectors.
    Supported: 'yolo', 'rtdetr' (via YOLOUltra)
    """
    if name.lower() in ['yolo', 'rtdetr', 'ultralytics']:
        return YOLOUltraDetector(
            model_path=cfg.get('model_path'),
            conf_threshold=cfg.get('conf', 0.25)
        )
    else:
        raise ValueError(f"Unknown detector type: {name}")
