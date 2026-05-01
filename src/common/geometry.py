from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def joint_angle_degrees(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denominator <= 1e-6:
        return 180.0
    cosine = float(np.dot(v1, v2) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def angle_degrees(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denominator = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denominator <= 1e-6:
        return 180.0
    cosine = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / denominator))
    return math.degrees(math.acos(cosine))


def normalize_skeleton_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize a COCO skeleton sequence shaped (T, V, C), with C=(x, y, conf)."""
    normalized = sequence.copy()
    if normalized.ndim != 3:
        return normalized
    _, joints, channels = normalized.shape
    if joints < 13 or channels < 3:
        return normalized

    for frame_id in range(normalized.shape[0]):
        hip_center = (normalized[frame_id, 11, :2] + normalized[frame_id, 12, :2]) / 2.0
        if np.allclose(hip_center, 0.0):
            hip_center = (normalized[frame_id, 5, :2] + normalized[frame_id, 6, :2]) / 2.0
        normalized[frame_id, :, :2] -= hip_center
        visible_y = normalized[frame_id, :, 1][sequence[frame_id, :, 2] > 0.1]
        if visible_y.size:
            height = float(np.max(visible_y) - np.min(visible_y))
            if height > 1e-6:
                normalized[frame_id, :, :2] /= height
    return normalized

