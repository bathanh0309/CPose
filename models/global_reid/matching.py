"""TFCS-PAR matching and score fusion helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_WEIGHTS = {
    "normal": {"face": 0.30, "body": 0.20, "pose": 0.15, "height": 0.10, "time": 0.15, "topology": 0.10},
    "no_face": {"body": 0.30, "pose": 0.20, "height": 0.15, "time": 0.20, "topology": 0.15},
    "clothing_change_suspected": {"face": 0.35, "body": 0.05, "pose": 0.20, "height": 0.15, "time": 0.15, "topology": 0.10},
}


@dataclass(slots=True)
class CandidateScores:
    score_total: float | None
    score_face: float | None
    score_body: float | None
    score_pose: float | None
    score_height: float | None
    score_time: float | None
    score_topology: float | None
    topology_allowed: bool | None
    delta_time_sec: float | None
    entry_zone: str | None
    exit_zone: str | None
    failure_reason: str


def weighted_fusion(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    usable = {key: value for key, value in scores.items() if value is not None and key in weights}
    if not usable:
        return None
    weight_total = sum(float(weights[key]) for key in usable)
    if weight_total <= 0:
        return None
    return sum(float(value) * float(weights[key]) / weight_total for key, value in usable.items())


def _safe_crop(frame: Any, bbox: list[float]) -> Any | None:
    if frame is None or bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def extract_body_hsv_feature(frame: Any, bbox: list[float]) -> np.ndarray | None:
    crop = _safe_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_height_ratio(bbox: list[float], frame_height: int | None = None) -> float | None:
    if not bbox or len(bbox) < 4:
        return None
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    if frame_height and frame_height > 0:
        return height / frame_height
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    return height / width if width > 0 else None


def cosine_similarity(a: list[float] | np.ndarray | None, b: list[float] | np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.shape != vb.shape or va.size == 0:
        return None
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return None
    return float((np.dot(va, vb) / denom + 1.0) / 2.0)


def histogram_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    value = float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def pose_signature(keypoints: list[dict] | None, bbox: list[float]) -> list[float] | None:
    if not keypoints or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    values: list[float] = []
    for keypoint in keypoints:
        conf = float(keypoint.get("confidence", 0.0))
        x = float(keypoint.get("x", 0.0))
        y = float(keypoint.get("y", 0.0))
        if conf < 0.3 or x <= 1.0 or y <= 1.0:
            values.extend([0.0, 0.0])
        else:
            values.extend([(x - x1) / width, (y - y1) / height])
    return values


__all__ = ["CandidateScores", "DEFAULT_WEIGHTS", "cosine_similarity", "extract_body_hsv_feature", "extract_height_ratio", "histogram_similarity", "pose_signature", "weighted_fusion"]
