from __future__ import annotations

from pathlib import Path

from src.common.config import get_research_section
from src.common.paths import MODELS_DIR, resolve_path


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def resolve_pose_model(model: str | Path | None = None) -> str:
    candidates = []
    if model:
        candidates.append(resolve_path(model))
    phase3 = get_research_section("phase3")
    models = get_research_section("models")
    if phase3.get("model"):
        candidates.append(resolve_path(phase3["model"]))
    if models.get("pose_model_path"):
        candidates.append(resolve_path(models["pose_model_path"]))
    candidates.extend([MODELS_DIR / "yolo" / "yolov8n-pose.pt", MODELS_DIR / "yolo" / "yolo11n-pose.pt", Path("yolov8n-pose.pt"), Path("yolo11n-pose.pt")])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    fallback = str(candidates[1])
    print(f"[WARN] local pose model not found, falling back to '{fallback}'.")
    return fallback
