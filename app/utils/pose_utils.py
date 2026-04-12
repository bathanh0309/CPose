"""Pose helpers for skeleton rendering and rule-based ADL classification."""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

SKELETON_EDGES = [
    (KP_NOSE, KP_LEFT_EYE),
    (KP_NOSE, KP_RIGHT_EYE),
    (KP_LEFT_EYE, KP_LEFT_EAR),
    (KP_RIGHT_EYE, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_LEFT_KNEE),
    (KP_LEFT_KNEE, KP_LEFT_ANKLE),
    (KP_RIGHT_HIP, KP_RIGHT_KNEE),
    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
]


def calc_angle(p1, vertex, p2) -> float:
    """Return the angle in degrees at vertex."""
    a = np.asarray(p1, dtype=float) - np.asarray(vertex, dtype=float)
    b = np.asarray(p2, dtype=float) - np.asarray(vertex, dtype=float)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 180.0
    cosine = float(np.clip(np.dot(a, b) / norm_product, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def calc_velocity(positions: Iterable[np.ndarray]) -> float:
    """Return the average displacement between consecutive positions."""
    points = [np.asarray(point, dtype=float) for point in positions if point is not None]
    if len(points) < 2:
        return 0.0
    distances = [float(np.linalg.norm(curr - prev)) for prev, curr in zip(points, points[1:])]
    return float(np.mean(distances)) if distances else 0.0


def draw_skeleton(frame, keypoints_xy, keypoints_conf, min_conf: float = 0.3):
    """Draw a COCO-17 skeleton overlay and return a new image."""
    output = frame.copy()
    xy = np.asarray(keypoints_xy, dtype=float)
    conf = np.asarray(keypoints_conf, dtype=float)

    for start_idx, end_idx in SKELETON_EDGES:
        if conf[start_idx] < min_conf or conf[end_idx] < min_conf:
            continue
        start = tuple(int(value) for value in xy[start_idx])
        end = tuple(int(value) for value in xy[end_idx])
        cv2.line(output, start, end, (72, 214, 171), 2)

    for idx, point in enumerate(xy):
        if conf[idx] < min_conf:
            continue
        center = tuple(int(value) for value in point)
        cv2.circle(output, center, 4, (255, 255, 255), -1)
        cv2.circle(output, center, 5, (56, 139, 253), 1)

    return output


def rule_based_adl(window, config) -> tuple[str, float]:
    """Classify ADL using the latest keypoints and short motion history."""
    if not window:
        return "unknown", 0.0

    thresholds = (config or {}).get("thresholds", {})
    min_conf = float((config or {}).get("keypoint_conf_min", 0.3))
    knee_bend_angle = float(thresholds.get("knee_bend_angle", 150))
    shoulder_raise = float(thresholds.get("shoulder_raise", 45))
    velocity_walk = float(thresholds.get("velocity_walk", 8.0))

    latest_xy, latest_conf = window[-1]
    latest_xy = np.asarray(latest_xy, dtype=float)
    latest_conf = np.asarray(latest_conf, dtype=float)
    visible_keypoints = int(np.sum(latest_conf >= min_conf))
    if visible_keypoints < 8:
        return "unknown", 0.2

    left_shoulder = _point(latest_xy, latest_conf, KP_LEFT_SHOULDER, min_conf)
    right_shoulder = _point(latest_xy, latest_conf, KP_RIGHT_SHOULDER, min_conf)
    left_hip = _point(latest_xy, latest_conf, KP_LEFT_HIP, min_conf)
    right_hip = _point(latest_xy, latest_conf, KP_RIGHT_HIP, min_conf)
    left_knee = _point(latest_xy, latest_conf, KP_LEFT_KNEE, min_conf)
    right_knee = _point(latest_xy, latest_conf, KP_RIGHT_KNEE, min_conf)
    left_ankle = _point(latest_xy, latest_conf, KP_LEFT_ANKLE, min_conf)
    right_ankle = _point(latest_xy, latest_conf, KP_RIGHT_ANKLE, min_conf)
    left_wrist = _point(latest_xy, latest_conf, KP_LEFT_WRIST, min_conf)
    right_wrist = _point(latest_xy, latest_conf, KP_RIGHT_WRIST, min_conf)

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)

    torso_angle = _torso_angle(shoulder_mid, hip_mid)
    knee_angles = []
    if left_hip is not None and left_knee is not None and left_ankle is not None:
        knee_angles.append(calc_angle(left_hip, left_knee, left_ankle))
    if right_hip is not None and right_knee is not None and right_ankle is not None:
        knee_angles.append(calc_angle(right_hip, right_knee, right_ankle))
    avg_knee_angle = float(np.mean(knee_angles)) if knee_angles else 180.0

    points = [point for point in [left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle] if point is not None]
    bbox_width, bbox_height = _bbox_size(points)
    aspect_ratio = bbox_width / bbox_height if bbox_height else 0.0

    ankle_positions = []
    for frame_xy, frame_conf in window:
        left = _point(frame_xy, frame_conf, KP_LEFT_ANKLE, min_conf)
        right = _point(frame_xy, frame_conf, KP_RIGHT_ANKLE, min_conf)
        ankle_positions.append(_midpoint(left, right))
    walk_velocity = calc_velocity(ankle_positions)

    wrists_above_shoulders = any(
        wrist is not None and shoulder is not None and wrist[1] < shoulder[1]
        for wrist, shoulder in [(left_wrist, left_shoulder), (right_wrist, right_shoulder)]
    )

    if torso_angle > 68 and walk_velocity > velocity_walk * 1.1:
        return "falling", 0.88
    if torso_angle > 68 and aspect_ratio > 1.15:
        return "lying_down", 0.84
    if avg_knee_angle < knee_bend_angle and walk_velocity < velocity_walk:
        return "sitting", 0.82
    if torso_angle > shoulder_raise and walk_velocity < velocity_walk * 0.6:
        return "bending", 0.78
    if wrists_above_shoulders:
        return "reaching", 0.76
    if walk_velocity > velocity_walk:
        return "walking", 0.79
    return "standing", 0.75


def _point(keypoints_xy, keypoints_conf, index: int, min_conf: float):
    conf = float(np.asarray(keypoints_conf)[index])
    if conf < min_conf:
        return None
    return np.asarray(keypoints_xy[index], dtype=float)


def _midpoint(p1, p2):
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return np.asarray(p2, dtype=float)
    if p2 is None:
        return np.asarray(p1, dtype=float)
    return (np.asarray(p1, dtype=float) + np.asarray(p2, dtype=float)) / 2.0


def _bbox_size(points: list[np.ndarray]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return max(xs) - min(xs), max(ys) - min(ys)


def _torso_angle(shoulder_mid, hip_mid) -> float:
    if shoulder_mid is None or hip_mid is None:
        return 0.0
    dx = abs(float(shoulder_mid[0] - hip_mid[0]))
    dy = abs(float(shoulder_mid[1] - hip_mid[1]))
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dx, dy + 1e-6))
