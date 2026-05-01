from __future__ import annotations

from pathlib import Path

from src.common.config import get_research_section
from src.common.paths import MODELS_DIR, resolve_path


def resolve_detection_model(model: str | Path | None = None) -> str:
    candidates = []
    if model:
        candidates.append(resolve_path(model))
    phase2 = get_research_section("phase2")
    models = get_research_section("models")
    if phase2.get("model"):
        candidates.append(resolve_path(phase2["model"]))
    if models.get("detector_model_path"):
        candidates.append(resolve_path(models["detector_model_path"]))
    candidates.extend([MODELS_DIR / "yolo" / "yolov8n.pt", MODELS_DIR / "yolo" / "yolo11n.pt", Path("yolov8n.pt"), Path("yolo11n.pt")])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    fallback = str(candidates[1])
    print(f"[WARN] local detection model not found, falling back to '{fallback}'.")
    return fallback
