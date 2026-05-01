from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.model_provider import get_yolo_model
from src.pose_estimation.config import COCO_KEYPOINT_NAMES


class PoseModel:
    def __init__(self, model_path: str | Path, conf: float = 0.5) -> None:
        self.model = get_yolo_model(str(model_path))
        self.conf = conf

    def estimate(self, frame: Any) -> list[dict]:
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        persons: list[dict] = []
        if not results:
            return persons
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None:
            return persons
        xy_data = keypoints.xy.detach().cpu().tolist()
        conf_data = keypoints.conf.detach().cpu().tolist() if keypoints.conf is not None else []
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            if class_id != 0:
                continue
            bbox = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
            person_keypoints = []
            for keypoint_id, point in enumerate(xy_data[index]):
                confidence = float(conf_data[index][keypoint_id]) if conf_data else 0.0
                person_keypoints.append({
                    "id": keypoint_id,
                    "name": COCO_KEYPOINT_NAMES[keypoint_id] if keypoint_id < len(COCO_KEYPOINT_NAMES) else str(keypoint_id),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "confidence": confidence,
                })
            persons.append({
                "track_id": None,
                "bbox": bbox,
                "keypoints": person_keypoints,
                "is_confirmed": False,
                "failure_reason": "NO_MATCHED_TRACK",
            })
        return persons
