from __future__ import annotations

import numpy as np


def normalize_skeleton(feat: np.ndarray) -> np.ndarray:
    """Normalize a COCO skeleton sequence shaped (T, V, C), with C=(x, y, conf)."""
    normalized = feat.copy()
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
        visible_y = normalized[frame_id, :, 1][feat[frame_id, :, 2] > 0.1]
        if visible_y.size:
            height = float(np.max(visible_y) - np.min(visible_y))
            if height > 1e-6:
                normalized[frame_id, :, :2] /= height
    return normalized

