from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _safe_crop(frame: Any, bbox: list[float]) -> Any | None:
    if frame is None or bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


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
