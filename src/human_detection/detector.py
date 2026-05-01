from __future__ import annotations

from typing import Any

from pathlib import Path

from src.common.model_provider import get_yolo_model


class PersonDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.5) -> None:
        self.model = get_yolo_model(str(model_path))
        self.conf = conf

    def detect(self, frame: Any) -> list[dict]:
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        detections: list[dict] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            if class_id == 0:
                detections.append({
                    "bbox": [float(v) for v in xyxy],
                    "confidence": confidence,
                    "class_id": 0,
                    "class_name": "person",
                    "failure_reason": "OK",
                })
        return detections
