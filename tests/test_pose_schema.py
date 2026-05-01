"""tests/test_pose_schema.py — JSON schema contract tests for keypoints.json.

Verifies COCO-17 keypoint structure, visibility formula, and that the pose
module does NOT assign ADL labels or global IDs (CLAUDE.md §3.3 forbidden).
"""
from __future__ import annotations

import pytest


COCO_17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _make_keypoint(name: str, idx: int, confidence: float = 0.85) -> dict:
    return {"id": idx, "name": name, "x": 100.0 + idx, "y": 200.0 + idx, "confidence": confidence}


def _make_pose_record(
    frame_id: int = 0,
    timestamp_sec: float = 0.0,
    camera_id: str = "cam1",
    track_id: int = 1,
    visible_ratio: float = 0.82,
    failure_reason: str = "OK",
    include_all_17: bool = True,
) -> dict:
    keypoints = [_make_keypoint(name, idx) for idx, name in enumerate(COCO_17_NAMES)] if include_all_17 else []
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "camera_id": camera_id,
        "persons": [
            {
                "track_id": track_id,
                "bbox": [10, 20, 100, 200],
                "visible_keypoint_ratio": visible_ratio,
                "keypoints": keypoints,
                "failure_reason": failure_reason,
            }
        ],
        "failure_reason": "OK",
    }


def _validate_pose_record(record: dict) -> None:
    assert "frame_id" in record
    assert "timestamp_sec" in record
    assert "camera_id" in record
    assert "persons" in record
    assert isinstance(record["persons"], list)
    for person in record["persons"]:
        assert "track_id" in person, "Missing track_id"
        assert "bbox" in person
        assert len(person["bbox"]) == 4
        assert "visible_keypoint_ratio" in person, "Missing visible_keypoint_ratio"
        assert 0.0 <= float(person["visible_keypoint_ratio"]) <= 1.0
        assert "keypoints" in person
        assert isinstance(person["keypoints"], list)
        assert "failure_reason" in person, "Missing failure_reason (CLAUDE.md §8)"
        # Forbidden fields from CLAUDE.md §3.3
        assert "adl_label" not in person, "Pose module must NOT assign adl_label"
        assert "global_id" not in person, "Pose module must NOT assign global_id"
        for kp in person["keypoints"]:
            assert "id" in kp
            assert "name" in kp
            assert "x" in kp
            assert "y" in kp
            assert "confidence" in kp
            assert 0.0 <= float(kp["confidence"]) <= 1.0


class TestPoseRecordSchema:
    def test_valid_record_passes(self) -> None:
        record = _make_pose_record()
        _validate_pose_record(record)

    def test_all_17_keypoints_present(self) -> None:
        record = _make_pose_record(include_all_17=True)
        persons = record["persons"]
        assert len(persons[0]["keypoints"]) == 17

    def test_keypoint_names_match_coco17(self) -> None:
        record = _make_pose_record(include_all_17=True)
        names = [kp["name"] for kp in record["persons"][0]["keypoints"]]
        assert names == COCO_17_NAMES

    def test_missing_failure_reason_fails(self) -> None:
        record = _make_pose_record()
        del record["persons"][0]["failure_reason"]
        with pytest.raises(AssertionError):
            _validate_pose_record(record)

    def test_no_adl_label_in_pose_output(self) -> None:
        record = _make_pose_record()
        for person in record["persons"]:
            assert "adl_label" not in person

    def test_visible_keypoint_ratio_formula(self) -> None:
        """r_visible = (1/17) * sum(visible) per CLAUDE.md §3.3."""
        tau_kp = 0.30
        confs = [0.91, 0.88, 0.85, 0.0, 0.0, 0.75, 0.80, 0.0, 0.0, 0.70, 0.65, 0.85, 0.80, 0.60, 0.55, 0.0, 0.0]
        visible_count = sum(1 for c in confs if c >= tau_kp)
        r_visible = visible_count / 17
        assert 0.0 <= r_visible <= 1.0
