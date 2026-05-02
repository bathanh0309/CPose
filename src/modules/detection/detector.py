"""YOLO person detector wrapper for CPose detection."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from src.common.config import get_research_section
from src.common.paths import MODELS_DIR, resolve_path


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


def resolve_detection_model(model: str | Path | None = None) -> str:
    candidates: list[Path] = []
    if model:
        candidates.append(resolve_path(model))
    phase2 = get_research_section("phase2")
    models = get_research_section("models")
    if phase2.get("model"):
        candidates.append(resolve_path(phase2["model"]))
    if models.get("detector_model_path"):
        candidates.append(resolve_path(models["detector_model_path"]))
    candidates.extend([
        MODELS_DIR / "yolo" / "yolov8n.pt",
        MODELS_DIR / "yolo" / "yolo11n.pt",
        Path("yolov8n.pt"),
        Path("yolo11n.pt"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    fallback = str(candidates[0] if candidates else Path("yolov8n.pt"))
    print(f"[WARN] local detection model not found, falling back to '{fallback}'.")
    return fallback


class PersonDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.5) -> None:
        self.model = _get_yolo_model(str(model_path))
        self.conf = conf

    def detect(self, frame: Any) -> list[dict]:
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        detections: list[dict] = []
        if not results or results[0].boxes is None:
            return detections
        for box in results[0].boxes:
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            if class_id != 0:
                continue
            detections.append({
                "bbox": [float(v) for v in box.xyxy[0].detach().cpu().tolist()],
                "confidence": float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0,
                "class_id": 0,
                "class_name": "person",
                "failure_reason": "OK",
            })
        return detections


__all__ = ["PersonDetector", "resolve_detection_model"]
