"""Pose model wrapper for CPose Module 3."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from src.common.config import get_research_section
from src.common.paths import MODELS_DIR, resolve_path


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}


def _get_yolo_model(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install requirements.txt") from exc
    if model_path not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if model_path not in _MODEL_CACHE:
                _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def resolve_pose_model(model: str | Path | None = None) -> str:
    candidates: list[Path] = []
    if model:
        candidates.append(resolve_path(model))
    phase3 = get_research_section("phase3")
    models = get_research_section("models")
    if phase3.get("model"):
        candidates.append(resolve_path(phase3["model"]))
    if models.get("pose_model_path"):
        candidates.append(resolve_path(models["pose_model_path"]))
    candidates.extend([
        MODELS_DIR / "yolo" / "yolov8n-pose.pt",
        MODELS_DIR / "yolo" / "yolo11n-pose.pt",
        Path("yolov8n-pose.pt"),
        Path("yolo11n-pose.pt"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    fallback = str(candidates[0] if candidates else Path("yolov8n-pose.pt"))
    print(f"[WARN] local pose model not found, falling back to '{fallback}'.")
    return fallback


class PoseModel:
    def __init__(self, model_path: str | Path, conf: float = 0.5, keypoint_conf: float = 0.30, min_visible_keypoints: int = 8) -> None:
        self.model = _get_yolo_model(str(model_path))
        self.conf = conf
        self.keypoint_conf = keypoint_conf
        self.min_visible_keypoints = min_visible_keypoints

    def estimate(self, frame: Any) -> list[dict]:
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        persons: list[dict] = []
        if not results:
            return persons
        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return persons
        xy_data = result.keypoints.xy.detach().cpu().tolist()
        conf_data = result.keypoints.conf.detach().cpu().tolist() if result.keypoints.conf is not None else []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            if class_id != 0:
                continue
            keypoints = []
            for keypoint_id, point in enumerate(xy_data[index]):
                confidence = float(conf_data[index][keypoint_id]) if conf_data else 0.0
                keypoints.append({"id": keypoint_id, "name": COCO_KEYPOINT_NAMES[keypoint_id] if keypoint_id < len(COCO_KEYPOINT_NAMES) else str(keypoint_id), "x": float(point[0]), "y": float(point[1]), "confidence": confidence})
            visible_count = sum(
                1
                for item in keypoints
                if (
                    float(item["confidence"]) >= self.keypoint_conf
                    and float(item["x"]) > 1.0
                    and float(item["y"]) > 1.0
                )
            )
            persons.append({
                "track_id": None,
                "bbox": [float(v) for v in box.xyxy[0].detach().cpu().tolist()],
                "bbox_confidence": float(box.conf[0].detach().cpu().item()) if box.conf is not None else None,
                "keypoints": keypoints,
                "is_confirmed": False,
                "visible_keypoint_count": visible_count,
                "visible_keypoint_ratio": visible_count / len(keypoints) if keypoints else 0.0,
                "pose_track_iou": None,
                "failure_reason": "OK" if visible_count >= self.min_visible_keypoints else "LOW_KEYPOINT_VISIBILITY",
            })
        return persons


__all__ = ["COCO_KEYPOINT_NAMES", "PoseModel", "resolve_pose_model"]
