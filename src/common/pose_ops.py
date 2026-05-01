from __future__ import annotations

import numpy as np

from src.common.geometry import joint_angle_degrees


def calculate_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return joint_angle_degrees(p1, p2, p3)


def get_posture_class(keypoints: np.ndarray) -> str:
    return "unknown"

