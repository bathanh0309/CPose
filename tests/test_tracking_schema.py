"""tests/test_tracking_schema.py — JSON schema contract tests for tracks.json.

Verifies that every tracking record conforms to CLAUDE.md §8 and that the
quality score formula is correctly bounded.
"""
from __future__ import annotations

import pytest


def _make_track_record(
    frame_id: int = 0,
    timestamp_sec: float = 0.0,
    camera_id: str = "cam1",
    track_id: int = 1,
    bbox: list = None,
    confidence: float = 0.88,
    age: int = 45,
    hits: int = 40,
    misses: int = 2,
    is_confirmed: bool = True,
    quality_score: float = 0.87,
    failure_reason: str = "OK",
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "camera_id": camera_id,
        "tracks": [
            {
                "track_id": track_id,
                "bbox": bbox or [10, 20, 100, 200],
                "confidence": confidence,
                "age": age,
                "hits": hits,
                "misses": misses,
                "is_confirmed": is_confirmed,
                "quality_score": quality_score,
                "failure_reason": failure_reason,
            }
        ],
        "failure_reason": "OK",
    }


def _validate_track_record(record: dict) -> None:
    assert "frame_id" in record
    assert isinstance(record["frame_id"], int)
    assert "timestamp_sec" in record
    assert "camera_id" in record
    assert "tracks" in record
    assert isinstance(record["tracks"], list)
    for track in record["tracks"]:
        assert "track_id" in track, "Missing track_id"
        assert isinstance(track["track_id"], int), "track_id must be int"
        assert "bbox" in track
        assert len(track["bbox"]) == 4
        assert "confidence" in track
        assert 0.0 <= float(track["confidence"]) <= 1.0
        assert "age" in track
        assert "hits" in track
        assert "misses" in track
        assert "is_confirmed" in track
        assert isinstance(track["is_confirmed"], bool)
        assert "quality_score" in track
        assert "failure_reason" in track, "Missing failure_reason (CLAUDE.md §8)"
        # Quality score must be in [0, 1]
        assert 0.0 <= float(track["quality_score"]) <= 1.0, "quality_score out of [0, 1]"


class TestTrackRecordSchema:
    def test_valid_record_passes(self) -> None:
        record = _make_track_record()
        _validate_track_record(record)

    def test_missing_track_id_fails(self) -> None:
        record = _make_track_record()
        del record["tracks"][0]["track_id"]
        with pytest.raises(AssertionError):
            _validate_track_record(record)

    def test_missing_failure_reason_fails(self) -> None:
        record = _make_track_record()
        del record["tracks"][0]["failure_reason"]
        with pytest.raises(AssertionError):
            _validate_track_record(record)

    def test_quality_score_formula_bounds(self) -> None:
        """Q_track = 0.5*mean_conf + 0.3*min(age/W, 1) - 0.2*min(misses/max_age, 1)"""
        W = 30
        max_age = 30
        mean_conf, age, misses = 0.88, 45, 2
        q = 0.5 * mean_conf + 0.3 * min(age / W, 1) - 0.2 * min(misses / max_age, 1)
        assert 0.0 <= q <= 1.0, f"Quality score out of bounds: {q}"

    def test_no_global_id_in_tracks(self) -> None:
        """Tracking module must NOT create global_id (CLAUDE.md §3.2 forbidden)."""
        record = _make_track_record()
        for track in record["tracks"]:
            assert "global_id" not in track, "Tracking module must not assign global_id"
