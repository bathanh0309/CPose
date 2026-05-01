from __future__ import annotations

import math
from collections import deque

from src.common.geometry import angle_degrees


def _point(person: dict, keypoint_id: int) -> tuple[float, float, float]:
    for keypoint in person.get("keypoints", []):
        if int(keypoint["id"]) == keypoint_id:
            return float(keypoint["x"]), float(keypoint["y"]), float(keypoint.get("confidence", 0.0))
    return 0.0, 0.0, 0.0


def _mean_point(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    visible = [point for point in points if point[2] >= 0.25]
    if not visible:
        return 0.0, 0.0, 0.0
    return (
        sum(point[0] for point in visible) / len(visible),
        sum(point[1] for point in visible) / len(visible),
        sum(point[2] for point in visible) / len(visible),
    )


def classify_adl(person: dict, history: deque[dict]) -> tuple[str, float]:
    keypoints = person.get("keypoints", [])
    visible_ratio = sum(1 for item in keypoints if float(item.get("confidence", 0.0)) >= 0.25) / len(keypoints) if keypoints else 0.0
    if visible_ratio < 0.35:
        return "unknown", 0.35

    left_shoulder, right_shoulder = _point(person, 5), _point(person, 6)
    left_hip, right_hip = _point(person, 11), _point(person, 12)
    left_knee, right_knee = _point(person, 13), _point(person, 14)
    left_ankle, right_ankle = _point(person, 15), _point(person, 16)
    left_wrist, right_wrist = _point(person, 9), _point(person, 10)
    shoulder = _mean_point([left_shoulder, right_shoulder])
    hip = _mean_point([left_hip, right_hip])
    knee = _mean_point([left_knee, right_knee])
    ankle = _mean_point([left_ankle, right_ankle])

    bbox = person.get("bbox", [0.0, 0.0, 1.0, 1.0])
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    aspect_ratio = width / height

    torso_dx = hip[0] - shoulder[0]
    torso_dy = hip[1] - shoulder[1]
    torso_angle = abs(math.degrees(math.atan2(torso_dy, torso_dx)))
    horizontal_torso = torso_angle < 35 or torso_angle > 145
    torso_lean = 35 <= torso_angle <= 62 or 118 <= torso_angle <= 145
    knee_angle = angle_degrees((hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1]))
    wrist_above_shoulder = (left_wrist[2] >= 0.25 and left_wrist[1] < shoulder[1]) or (right_wrist[2] >= 0.25 and right_wrist[1] < shoulder[1])

    velocity = 0.0
    if history:
        previous = history[0]
        previous_ankle = previous.get("ankle")
        if previous_ankle:
            velocity = math.hypot(ankle[0] - previous_ankle[0], ankle[1] - previous_ankle[1]) / max(height, 1.0)

    if horizontal_torso and velocity > 0.08:
        return "falling", 0.78
    if horizontal_torso or aspect_ratio > 1.15:
        return "lying_down", 0.74
    if knee_angle < 125 and velocity < 0.035:
        return "sitting", 0.72
    if torso_lean and velocity < 0.05:
        return "bending", 0.68
    if wrist_above_shoulder:
        return "reaching", 0.66
    if velocity > 0.05:
        return "walking", 0.70
    return "standing", 0.62


def history_item(person: dict) -> dict:
    left_ankle, right_ankle = _point(person, 15), _point(person, 16)
    return {"ankle": _mean_point([left_ankle, right_ankle])}
