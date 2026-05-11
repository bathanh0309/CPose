from __future__ import annotations

from collections import deque

from models.adl_recognition.rule_based_adl import classify_adl


def _person(points: dict[int, tuple[float, float]], bbox: list[float]) -> dict:
    return {
        "is_confirmed": True,
        "visible_keypoint_count": 8,
        "visible_keypoint_ratio": 0.7,
        "bbox": bbox,
        "keypoints": [
            {"id": keypoint_id, "x": x, "y": y, "confidence": 0.9}
            for keypoint_id, (x, y) in points.items()
        ],
        "frame_height": 480,
    }


def test_standing_torso_angle_is_not_lying() -> None:
    person = _person(
        {
            5: (145, 150),
            6: (155, 150),
            11: (145, 280),
            12: (155, 280),
            13: (145, 330),
            14: (155, 330),
            15: (145, 390),
            16: (155, 390),
        },
        [100, 50, 200, 400],
    )

    label, _confidence, _reason = classify_adl(person, deque())

    assert label == "standing"


def test_horizontal_torso_classifies_as_lying_down() -> None:
    person = _person(
        {
            5: (100, 150),
            6: (110, 150),
            11: (220, 150),
            12: (230, 150),
            13: (280, 150),
            14: (290, 150),
            15: (340, 150),
            16: (350, 150),
        },
        [80, 120, 370, 210],
    )

    label, _confidence, _reason = classify_adl(person, deque())

    assert label == "lying_down"
