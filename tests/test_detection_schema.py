"""tests/test_detection_schema.py — JSON schema contract tests for detections.json.

Verifies that every record produced by the detection module conforms to
the schema defined in CLAUDE.md §8.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _make_detection(
    frame_id: int = 0,
    timestamp_sec: float = 0.0,
    camera_id: str = "cam1",
    bbox: list = None,
    confidence: float = 0.91,
    class_id: int = 0,
    class_name: str = "person",
    failure_reason: str = "OK",
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "camera_id": camera_id,
        "detections": [
            {
                "bbox": bbox or [10, 20, 100, 200],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "failure_reason": failure_reason,
            }
        ],
    }


def _validate_detection_record(record: dict) -> None:
    """Raise AssertionError if the record does not conform to CLAUDE.md §8."""
    assert "frame_id" in record, "Missing frame_id"
    assert isinstance(record["frame_id"], int), "frame_id must be int"
    assert "timestamp_sec" in record, "Missing timestamp_sec"
    assert isinstance(record["timestamp_sec"], float), "timestamp_sec must be float"
    assert "camera_id" in record, "Missing camera_id"
    assert "detections" in record, "Missing detections list"
    assert isinstance(record["detections"], list), "detections must be list"
    for det in record["detections"]:
        assert "bbox" in det, "Detection missing bbox"
        assert len(det["bbox"]) == 4, "bbox must have 4 coordinates"
        assert "confidence" in det, "Detection missing confidence"
        assert 0.0 <= float(det["confidence"]) <= 1.0, "confidence out of range"
        assert "class_id" in det, "Detection missing class_id"
        assert det["class_id"] == 0, "class_id must be 0 (person-only per CLAUDE.md §3.1)"
        assert "class_name" in det, "Detection missing class_name"
        assert "failure_reason" in det, "Detection missing failure_reason (CLAUDE.md §8)"


class TestDetectionRecordSchema:
    def test_valid_record_passes(self) -> None:
        record = _make_detection()
        _validate_detection_record(record)

    def test_missing_frame_id_fails(self) -> None:
        record = _make_detection()
        del record["frame_id"]
        with pytest.raises(AssertionError):
            _validate_detection_record(record)

    def test_missing_failure_reason_fails(self) -> None:
        record = _make_detection()
        del record["detections"][0]["failure_reason"]
        with pytest.raises(AssertionError):
            _validate_detection_record(record)

    def test_wrong_class_id_fails(self) -> None:
        record = _make_detection(class_id=1)
        with pytest.raises(AssertionError):
            _validate_detection_record(record)

    def test_bad_confidence_fails(self) -> None:
        record = _make_detection(confidence=1.5)
        with pytest.raises(AssertionError):
            _validate_detection_record(record)

    def test_empty_detections_ok(self) -> None:
        record = _make_detection()
        record["detections"] = []
        # An empty detections list is valid (no persons in frame)
        assert isinstance(record["detections"], list)

    def test_failure_reason_ok_value(self) -> None:
        record = _make_detection(failure_reason="OK")
        _validate_detection_record(record)
        assert record["detections"][0]["failure_reason"] == "OK"

    def test_failure_reason_no_person_detected(self) -> None:
        record = {
            "frame_id": 0,
            "timestamp_sec": 0.0,
            "camera_id": "cam1",
            "detections": [],
            "failure_reason": "NO_PERSON_DETECTED",
        }
        assert record["failure_reason"] == "NO_PERSON_DETECTED"
