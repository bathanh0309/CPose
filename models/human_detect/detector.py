"""YOLO person detector wrapper for CPose detection — enhanced with evidence fields."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_research_section
from src.paths import MODELS_DIR, resolve_path


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}
_DETECTION_COUNTER = 0
_DETECTION_LOCK = threading.Lock()


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


def _next_detection_id() -> str:
    global _DETECTION_COUNTER
    with _DETECTION_LOCK:
        _DETECTION_COUNTER += 1
        return f"D{_DETECTION_COUNTER:06d}"


def reset_detection_counter() -> None:
    global _DETECTION_COUNTER
    with _DETECTION_LOCK:
        _DETECTION_COUNTER = 0


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


def _compute_detection_quality(
    confidence: float,
    bbox: list[float],
    frame_w: int,
    frame_h: int,
    border_margin: float = 0.02,
) -> float:
    """Proxy detection quality from confidence, area, aspect ratio, and border distance."""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    frame_area = float(frame_w * frame_h) or 1.0

    # Normalised area — prefer persons that are reasonably sized (not tiny/huge)
    area_norm = area / frame_area
    area_score = 1.0 - abs(area_norm - 0.08) / 0.15  # peaks at ~8% of frame
    area_score = float(max(0.0, min(1.0, area_score)))

    # Aspect ratio — standing person ~0.35–0.55 wide/tall
    aspect = (w / h) if h > 0 else 0.0
    ideal_aspect = 0.42
    aspect_score = 1.0 - abs(aspect - ideal_aspect) / 0.40
    aspect_score = float(max(0.0, min(1.0, aspect_score)))

    # Border proximity penalty
    margin_x = border_margin * frame_w
    margin_y = border_margin * frame_h
    at_border = x1 < margin_x or y1 < margin_y or x2 > frame_w - margin_x or y2 > frame_h - margin_y
    border_penalty = 0.15 if at_border else 0.0

    quality = 0.55 * confidence + 0.25 * area_score + 0.20 * aspect_score - border_penalty
    return float(max(0.0, min(1.0, quality)))


class PersonDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.5) -> None:
        self.model = _get_yolo_model(str(model_path))
        self.conf = conf

    def detect(self, frame: Any, crops_dir: Path | None = None, frame_id: int = 0, camera_id: str = "") -> list[dict]:
        import cv2
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        detections: list[dict] = []
        if not results or results[0].boxes is None:
            return detections
        frame_h, frame_w = frame.shape[:2]
        for box in results[0].boxes:
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            if class_id != 0:
                continue
            bbox = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bcx = cx
            bcy = y2
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            aspect = (w / h) if h > 0 else 0.0
            quality = _compute_detection_quality(confidence, bbox, frame_w, frame_h)
            det_id = _next_detection_id()

            crop_path: str | None = None
            if crops_dir is not None:
                try:
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                    ix2, iy2 = min(frame_w, int(x2)), min(frame_h, int(y2))
                    if ix2 > ix1 and iy2 > iy1:
                        crop = frame[iy1:iy2, ix1:ix2]
                        crop_name = f"frame_{frame_id:06d}_{det_id}.jpg"
                        crop_file = crops_dir / crop_name
                        cv2.imwrite(str(crop_file), crop)
                        crop_path = str(crop_file.relative_to(crops_dir.parent.parent) if crops_dir.parent.parent.exists() else crop_file)
                except Exception:
                    crop_path = None

            detections.append({
                "detection_id": det_id,
                "bbox": bbox,
                "center": [round(cx, 1), round(cy, 1)],
                "bottom_center": [round(bcx, 1), round(bcy, 1)],
                "bbox_area": round(area, 1),
                "aspect_ratio": round(aspect, 3),
                "confidence": round(confidence, 4),
                "class_id": 0,
                "class_name": "person",
                "crop_path": crop_path,
                "detection_quality": round(quality, 4),
                "failure_reason": "OK",
            })
        return detections


__all__ = ["PersonDetector", "reset_detection_counter", "resolve_detection_model"]
