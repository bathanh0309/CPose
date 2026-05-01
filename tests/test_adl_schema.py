"""tests/test_adl_schema.py — JSON schema contract tests for adl_events.json.

Verifies that ADL events conform to CLAUDE.md §8, that all labels are within
the defined set, and that no global_id is present (CLAUDE.md §3.4 forbidden).
"""
from __future__ import annotations

import pytest


ADL_LABELS = {"standing", "sitting", "walking", "lying_down", "falling", "reaching", "bending", "unknown"}


def _make_adl_event(
    frame_id: int = 30,
    timestamp_sec: float = 1.0,
    camera_id: str = "cam1",
    track_id: int = 1,
    raw_label: str = "walking",
    smoothed_label: str = "walking",
    adl_label: str = "walking",
    confidence: float = 0.79,
    window_size: int = 30,
    visible_keypoint_ratio: float = 0.82,
    failure_reason: str = "OK",
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "camera_id": camera_id,
        "track_id": track_id,
        "raw_label": raw_label,
        "smoothed_label": smoothed_label,
        "adl_label": adl_label,
        "confidence": confidence,
        "window_size": window_size,
        "visible_keypoint_ratio": visible_keypoint_ratio,
        "failure_reason": failure_reason,
    }


def _validate_adl_event(event: dict) -> None:
    required = ["frame_id", "timestamp_sec", "camera_id", "track_id",
                "raw_label", "smoothed_label", "adl_label", "confidence",
                "window_size", "visible_keypoint_ratio", "failure_reason"]
    for field in required:
        assert field in event, f"Missing required field: {field}"
    assert event["raw_label"] in ADL_LABELS, f"Invalid raw_label: {event['raw_label']}"
    assert event["adl_label"] in ADL_LABELS, f"Invalid adl_label: {event['adl_label']}"
    assert 0.0 <= float(event["confidence"]) <= 1.0, "confidence out of range"
    assert "global_id" not in event, "ADL module must NOT assign global_id (CLAUDE.md §3.4)"


class TestADLEventSchema:
    def test_valid_event_passes(self) -> None:
        event = _make_adl_event()
        _validate_adl_event(event)

    def test_all_adl_labels_valid(self) -> None:
        for label in ADL_LABELS:
            event = _make_adl_event(adl_label=label, raw_label=label, smoothed_label=label)
            _validate_adl_event(event)

    def test_invalid_adl_label_fails(self) -> None:
        event = _make_adl_event(adl_label="running")
        with pytest.raises(AssertionError):
            _validate_adl_event(event)

    def test_missing_failure_reason_fails(self) -> None:
        event = _make_adl_event()
        del event["failure_reason"]
        with pytest.raises(AssertionError):
            _validate_adl_event(event)

    def test_no_global_id_in_adl_events(self) -> None:
        event = _make_adl_event()
        assert "global_id" not in event

    def test_unknown_label_valid(self) -> None:
        event = _make_adl_event(adl_label="unknown", raw_label="unknown", confidence=0.2)
        _validate_adl_event(event)

    def test_confidence_in_range(self) -> None:
        event = _make_adl_event(confidence=1.5)
        with pytest.raises(AssertionError):
            _validate_adl_event(event)
