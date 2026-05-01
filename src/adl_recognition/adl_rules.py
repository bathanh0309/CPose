from __future__ import annotations

import math
from collections import deque

from src.common.geometry import angle_degrees
from src.adl_recognition.config import ADLConfig


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


def classify_adl(person: dict, history: deque[dict], config: ADLConfig | None = None) -> tuple[str, float | None, str]:
    config = config or ADLConfig()
    if not bool(person.get("is_confirmed", True)):
        return "unknown", None, "UNCONFIRMED_TRACK"
    keypoints = person.get("keypoints", [])
    visible_count = int(person.get("visible_keypoint_count") or sum(1 for item in keypoints if float(item.get("confidence", 0.0)) >= 0.25))
    visible_ratio = float(person.get("visible_keypoint_ratio") or (visible_count / len(keypoints) if keypoints else 0.0))
    if visible_count < config.min_visible_keypoints or visible_ratio < config.min_visible_ratio:
        return "unknown", 0.2, "LOW_KEYPOINT_VISIBILITY"

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
    horizontal_torso = torso_angle < config.torso_horizontal_low or torso_angle > (180 - config.torso_horizontal_low)
    torso_lean = config.torso_lean_low <= torso_angle <= config.torso_lean_high or (180 - config.torso_lean_high) <= torso_angle <= (180 - config.torso_lean_low)
    knee_angle = angle_degrees((hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1]))
    wrist_above_shoulder = (left_wrist[2] >= 0.25 and left_wrist[1] < shoulder[1]) or (right_wrist[2] >= 0.25 and right_wrist[1] < shoulder[1])

    velocity = 0.0
    if history:
        previous = history[0]
        previous_ankle = previous.get("ankle")
        if previous_ankle:
            velocity = math.hypot(ankle[0] - previous_ankle[0], ankle[1] - previous_ankle[1]) / max(height, 1.0)

    reason = "OK" if len(history) >= min(3, config.window_size) else "SHORT_TRACK_WINDOW"
    if horizontal_torso and velocity > config.falling_velocity:
        return "falling", 0.78, reason
    if horizontal_torso or aspect_ratio > config.lying_aspect_ratio:
        return "lying_down", 0.74, reason
    if knee_angle < config.knee_sitting_angle and velocity < 0.035:
        return "sitting", 0.72, reason
    if torso_lean and velocity < config.walking_velocity:
        return "bending", 0.68, reason
    if wrist_above_shoulder:
        return "reaching", 0.66, reason
    if velocity > config.walking_velocity:
        return "walking", 0.70, reason
    return "standing", 0.62, reason


def history_item(person: dict) -> dict:
    left_ankle, right_ankle = _point(person, 15), _point(person, 16)
    return {"ankle": _mean_point([left_ankle, right_ankle])}
