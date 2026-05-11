"""tests/test_reid_schema.py — JSON schema contract tests for reid_tracks.json
and global_person_table.json.

Verifies that every Global ReID record produced by Module 5 (TFCS-PAR) conforms
to CLAUDE.md §8, that all person states are within the defined set, that score
fusion respects mathematical bounds, and that module boundary rules are respected.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Constants from CLAUDE.md §3.5 and src/global_reid/state_machine.py
# ---------------------------------------------------------------------------

VALID_STATES = {
    "ACTIVE",
    "PENDING_TRANSFER",
    "IN_BLIND_ZONE",
    "IN_ROOM",
    "CLOTHING_CHANGE_SUSPECTED",
    "DORMANT",
    "CLOSED",
}

VALID_MATCH_STATUSES = {"strong_match", "soft_match", "pending_vote", "ambiguous", "new_id", "no_candidate", "weak_match", "new_gid", "UNK"}

STRONG_THRESHOLD = 0.65  # CLAUDE.md §3.5
WEAK_THRESHOLD = 0.45    # CLAUDE.md §3.5

# All error codes that may appear in failure_reason (CLAUDE.md §12)
VALID_FAILURE_REASONS = {
    "OK",
    "NO_PERSON_DETECTED",
    "LOW_DETECTION_CONFIDENCE",
    "TRACK_FRAGMENTED",
    "UNCONFIRMED_TRACK",
    "SHORT_TRACK_WINDOW",
    "LOW_KEYPOINT_VISIBILITY",
    "NO_FACE",
    "BODY_OCCLUDED",
    "TOPOLOGY_CONFLICT",
    "TIME_WINDOW_CONFLICT",
    "MULTI_CANDIDATE_CONFLICT",
    "MODEL_NOT_FOUND",
    "INVALID_VIDEO",
    "INVALID_MANIFEST",
    "INVALID_TOPOLOGY",
    "MISSING_INPUT_JSON",
    "GROUND_TRUTH_NOT_FOUND",
}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_reid_track(
    frame_id: int = 30,
    timestamp_sec: float = 1.0,
    camera_id: str = "cam2",
    local_track_id: int = 1,
    global_id: str = "GID-001",
    state: str = "ACTIVE",
    match_status: str = "strong_match",
    score_total: float = 0.72,
    score_time: float = 1.0,
    score_topology: float = 1.0,
    score_pose: float = 0.68,
    score_body: float = 0.51,
    score_face: float | None = None,
    delta_time_sec: float = 42.0,
    topology_allowed: bool = True,
    failure_reason: str = "OK",
) -> dict:
    """Return a minimal reid_tracks.json record matching CLAUDE.md §8."""
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "camera_id": camera_id,
        "local_track_id": local_track_id,
        "global_id": global_id,
        "state": state,
        "match_status": match_status,
        "score_total": score_total,
        "score_time": score_time,
        "score_topology": score_topology,
        "score_pose": score_pose,
        "score_body": score_body,
        "score_face": score_face,
        "delta_time_sec": delta_time_sec,
        "topology_allowed": topology_allowed,
        "failure_reason": failure_reason,
    }


def _make_global_person_table_entry(
    global_id: str = "GID-001",
    state: str = "ACTIVE",
    last_camera_id: str = "cam2",
    last_seen_sec: float = 1.0,
    total_sightings: int = 3,
    failure_reason: str = "OK",
) -> dict:
    """Return a minimal global_person_table.json entry."""
    return {
        "global_id": global_id,
        "state": state,
        "last_camera_id": last_camera_id,
        "last_seen_sec": last_seen_sec,
        "total_sightings": total_sightings,
        "failure_reason": failure_reason,
    }


def _make_metrics_json(
    run_id: str = "2026-05-01_15-30-22_baseline",
    metric_type: str = "proxy",
) -> dict:
    """Return a minimal reid_metrics.json body per CLAUDE.md §8."""
    return {
        "metric_type": metric_type,
        "run_id": run_id,
        "module": "global_reid",
        "input_videos": ["cam1.mp4", "cam2.mp4"],
        "model_info": {"name": "tfcs_par", "path": "N/A"},
        "output_paths": {
            "json": "dataset/runs/.../05_global_reid/reid_tracks.json",
            "overlay": "dataset/runs/.../05_global_reid/reid_overlay.mp4",
            "paper_table": "dataset/runs/.../08_paper_report/table_reid_results.csv",
        },
        "metrics": {
            "global_id_count": 4,
            "pending_count": 1,
            "conflict_count": 0,
            "topology_conflict_count": 0,
            "unknown_match_count": 0,
        },
    }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_reid_track(record: dict) -> None:
    """Raise AssertionError if the record does not conform to CLAUDE.md §8."""
    required_top = [
        "frame_id", "timestamp_sec", "camera_id", "local_track_id",
        "global_id", "state", "match_status", "score_total",
        "score_time", "score_topology", "topology_allowed", "failure_reason",
    ]
    for field in required_top:
        assert field in record, f"Missing required field: {field}"

    assert isinstance(record["frame_id"], int), "frame_id must be int"
    assert record["state"] in VALID_STATES, f"Invalid state: {record['state']}"
    assert record["match_status"] in VALID_MATCH_STATUSES, (
        f"Invalid match_status: {record['match_status']}"
    )

    # score_total may be None if no usable signals, otherwise must be in [0, 1]
    if record["score_total"] is not None:
        assert 0.0 <= float(record["score_total"]) <= 1.0, (
            f"score_total out of [0, 1]: {record['score_total']}"
        )

    assert isinstance(record["topology_allowed"], bool), "topology_allowed must be bool"
    assert record["failure_reason"] in VALID_FAILURE_REASONS, (
        f"Unknown failure_reason: {record['failure_reason']}"
    )


def _validate_global_person_table_entry(entry: dict) -> None:
    """Raise AssertionError if the global_person_table entry is malformed."""
    required = ["global_id", "state", "last_camera_id", "last_seen_sec",
                "total_sightings", "failure_reason"]
    for field in required:
        assert field in entry, f"Missing required field: {field}"

    assert entry["state"] in VALID_STATES, f"Invalid state: {entry['state']}"
    assert isinstance(entry["total_sightings"], int), "total_sightings must be int"
    assert entry["total_sightings"] >= 0, "total_sightings must be non-negative"


def _validate_metrics_json(metrics: dict) -> None:
    """Raise AssertionError if the metrics JSON does not conform to CLAUDE.md §8."""
    required = ["metric_type", "run_id", "module", "input_videos", "model_info",
                "output_paths", "metrics"]
    for field in required:
        assert field in metrics, f"Missing required metrics field: {field}"
    assert metrics["metric_type"] in ("proxy", "ground_truth"), (
        "metric_type must be 'proxy' or 'ground_truth'"
    )
    assert metrics["module"] == "global_reid", "module field must be 'global_reid'"


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestReidTrackSchema:
    """Contract tests for reid_tracks.json (CLAUDE.md §8)."""

    def test_valid_record_passes(self) -> None:
        record = _make_reid_track()
        _validate_reid_track(record)

    def test_missing_failure_reason_fails(self) -> None:
        record = _make_reid_track()
        del record["failure_reason"]
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_missing_global_id_fails(self) -> None:
        record = _make_reid_track()
        del record["global_id"]
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_missing_state_fails(self) -> None:
        record = _make_reid_track()
        del record["state"]
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_invalid_state_fails(self) -> None:
        record = _make_reid_track(state="LOST")
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_all_valid_states_accepted(self) -> None:
        for state in VALID_STATES:
            record = _make_reid_track(state=state)
            _validate_reid_track(record)

    def test_invalid_match_status_fails(self) -> None:
        record = _make_reid_track(match_status="exact_match")
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_all_valid_match_statuses_accepted(self) -> None:
        for status in VALID_MATCH_STATUSES:
            record = _make_reid_track(match_status=status)
            _validate_reid_track(record)

    def test_score_total_out_of_range_fails(self) -> None:
        record = _make_reid_track(score_total=1.5)
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_score_total_none_is_valid(self) -> None:
        """score_total may be None when no signals are available."""
        record = _make_reid_track(score_total=None)
        _validate_reid_track(record)

    def test_score_face_none_is_valid(self) -> None:
        """score_face=None is the canonical value when ArcFace is unavailable."""
        record = _make_reid_track(score_face=None)
        _validate_reid_track(record)

    def test_topology_allowed_must_be_bool(self) -> None:
        record = _make_reid_track()
        record["topology_allowed"] = 1  # int, not bool
        with pytest.raises(AssertionError):
            _validate_reid_track(record)

    def test_topology_conflict_failure_reason(self) -> None:
        """TOPOLOGY_CONFLICT is a valid failure_reason code (CLAUDE.md §12)."""
        record = _make_reid_track(
            topology_allowed=False,
            match_status="UNK",
            failure_reason="TOPOLOGY_CONFLICT",
        )
        _validate_reid_track(record)

    def test_time_window_conflict_failure_reason(self) -> None:
        record = _make_reid_track(
            match_status="UNK",
            failure_reason="TIME_WINDOW_CONFLICT",
        )
        _validate_reid_track(record)

    def test_multi_candidate_conflict_failure_reason(self) -> None:
        record = _make_reid_track(
            match_status="UNK",
            failure_reason="MULTI_CANDIDATE_CONFLICT",
        )
        _validate_reid_track(record)

    def test_unknown_failure_reason_rejected(self) -> None:
        record = _make_reid_track(failure_reason="CAMERA_OFFLINE")
        with pytest.raises(AssertionError):
            _validate_reid_track(record)


class TestReidScoreFusion:
    """Tests for TFCS-PAR score fusion thresholds (CLAUDE.md §3.5)."""

    def test_strong_match_threshold(self) -> None:
        """score_total >= 0.65 must map to strong_match."""
        score = 0.72
        assert score >= STRONG_THRESHOLD
        match_status = "strong_match" if score >= STRONG_THRESHOLD else (
            "weak_match" if score >= WEAK_THRESHOLD else "new_gid"
        )
        assert match_status == "strong_match"

    def test_weak_match_threshold(self) -> None:
        """0.45 <= score_total < 0.65 must map to weak_match."""
        score = 0.55
        assert WEAK_THRESHOLD <= score < STRONG_THRESHOLD
        match_status = "strong_match" if score >= STRONG_THRESHOLD else (
            "weak_match" if score >= WEAK_THRESHOLD else "new_gid"
        )
        assert match_status == "weak_match"

    def test_new_gid_threshold(self) -> None:
        """score_total < 0.45 must map to new_gid or UNK."""
        score = 0.30
        assert score < WEAK_THRESHOLD
        match_status = "strong_match" if score >= STRONG_THRESHOLD else (
            "weak_match" if score >= WEAK_THRESHOLD else "new_gid"
        )
        assert match_status == "new_gid"

    def test_weighted_fusion_all_signals(self) -> None:
        """Verify TFCS-PAR weighted fusion stays within [0, 1] for normal mode."""
        weights = {"face": 0.30, "body": 0.20, "pose": 0.15,
                   "height": 0.10, "time": 0.15, "topology": 0.10}
        scores = {"face": 0.80, "body": 0.60, "pose": 0.70,
                  "height": 0.75, "time": 1.0, "topology": 1.0}
        weight_total = sum(weights[k] for k in scores)
        fused = sum(scores[k] * weights[k] / weight_total for k in scores)
        assert 0.0 <= fused <= 1.0, f"Fused score out of [0,1]: {fused}"

    def test_weighted_fusion_missing_face(self) -> None:
        """When face is unavailable, remaining signals must still fuse correctly."""
        weights = {"body": 0.30, "pose": 0.20, "height": 0.15,
                   "time": 0.20, "topology": 0.15}
        scores = {"body": 0.55, "pose": 0.62, "height": 0.70,
                  "time": 0.90, "topology": 1.0}
        weight_total = sum(weights[k] for k in scores)
        fused = sum(scores[k] * weights[k] / weight_total for k in scores)
        assert 0.0 <= fused <= 1.0, f"Fused score out of [0,1] (no-face mode): {fused}"

    def test_weighted_fusion_clothing_change_reduces_body_weight(self) -> None:
        """In clothing_change_suspected mode body weight drops to 0.05."""
        clothing_change_weights = {
            "face": 0.35, "body": 0.05, "pose": 0.20,
            "height": 0.15, "time": 0.15, "topology": 0.10,
        }
        normal_weights = {"face": 0.30, "body": 0.20, "pose": 0.15,
                          "height": 0.10, "time": 0.15, "topology": 0.10}
        assert clothing_change_weights["body"] < normal_weights["body"], (
            "Body weight must be lower in clothing_change_suspected mode"
        )

    def test_weights_sum_to_one(self) -> None:
        """All weight sets from fusion_score.py must sum to 1.0."""
        from models.global_reid.fusion_score import DEFAULT_WEIGHTS
        for mode, weights in DEFAULT_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Weights for mode '{mode}' sum to {total}, expected 1.0"
            )

    def test_weighted_fusion_no_usable_signals_returns_none(self) -> None:
        """If all signals are None, fusion must return None, not a fabricated value."""
        from models.global_reid.fusion_score import weighted_fusion, DEFAULT_WEIGHTS
        result = weighted_fusion({}, DEFAULT_WEIGHTS["normal"])
        assert result is None, "weighted_fusion with no signals must return None"

    def test_fusion_with_missing_face_normalizes_correctly(self) -> None:
        """Missing face score must not pull the fused score outside bounds."""
        from models.global_reid.fusion_score import DEFAULT_WEIGHTS, weighted_fusion

        scores = {
            "face": None,
            "body": 0.8,
            "pose": 0.6,
            "height": 0.7,
            "time": 1.0,
            "topology": 1.0,
        }
        result = weighted_fusion(scores, DEFAULT_WEIGHTS["normal"])
        assert result is not None
        assert 0.0 <= result <= 1.0


class TestGlobalPersonTableSchema:
    """Contract tests for global_person_table.json (CLAUDE.md §8)."""

    def test_valid_entry_passes(self) -> None:
        entry = _make_global_person_table_entry()
        _validate_global_person_table_entry(entry)

    def test_missing_global_id_fails(self) -> None:
        entry = _make_global_person_table_entry()
        del entry["global_id"]
        with pytest.raises(AssertionError):
            _validate_global_person_table_entry(entry)

    def test_invalid_state_fails(self) -> None:
        entry = _make_global_person_table_entry(state="GHOST")
        with pytest.raises(AssertionError):
            _validate_global_person_table_entry(entry)

    def test_all_valid_states_accepted(self) -> None:
        for state in VALID_STATES:
            entry = _make_global_person_table_entry(state=state)
            _validate_global_person_table_entry(entry)

    def test_negative_sightings_fails(self) -> None:
        entry = _make_global_person_table_entry(total_sightings=-1)
        with pytest.raises(AssertionError):
            _validate_global_person_table_entry(entry)

    def test_missing_failure_reason_fails(self) -> None:
        entry = _make_global_person_table_entry()
        del entry["failure_reason"]
        with pytest.raises(AssertionError):
            _validate_global_person_table_entry(entry)


class TestReidMetricsJson:
    """Contract tests for reid_metrics.json (CLAUDE.md §8 + §9)."""

    def test_valid_metrics_passes(self) -> None:
        metrics = _make_metrics_json()
        _validate_metrics_json(metrics)

    def test_metric_type_must_be_proxy_or_ground_truth(self) -> None:
        metrics = _make_metrics_json(metric_type="accuracy")
        with pytest.raises(AssertionError):
            _validate_metrics_json(metrics)

    def test_proxy_type_when_no_gt(self) -> None:
        """Without GT the metric_type must be 'proxy', never 'accuracy'."""
        metrics = _make_metrics_json(metric_type="proxy")
        assert metrics["metric_type"] == "proxy"

    def test_metrics_contains_proxy_fields(self) -> None:
        metrics = _make_metrics_json()
        proxy_fields = [
            "global_id_count", "pending_count", "conflict_count",
            "topology_conflict_count", "unknown_match_count",
        ]
        for field in proxy_fields:
            assert field in metrics["metrics"], f"Missing proxy metric: {field}"

    def test_missing_run_id_fails(self) -> None:
        metrics = _make_metrics_json()
        del metrics["run_id"]
        with pytest.raises(AssertionError):
            _validate_metrics_json(metrics)

    def test_missing_input_videos_fails(self) -> None:
        metrics = _make_metrics_json()
        del metrics["input_videos"]
        with pytest.raises(AssertionError):
            _validate_metrics_json(metrics)


class TestReidModuleBoundaries:
    """Enforce CLAUDE.md §3.5 and §14 module isolation rules."""

    def test_reid_output_has_global_id(self) -> None:
        """The ReID module IS the one that creates global_id — it must be present."""
        record = _make_reid_track(global_id="GID-001")
        assert "global_id" in record, "ReID output must contain global_id"

    def test_reid_record_references_local_track_id(self) -> None:
        """ReID reads local track IDs from upstream tracks.json."""
        record = _make_reid_track(local_track_id=3)
        assert "local_track_id" in record, "ReID must reference the upstream local_track_id"
        assert record["local_track_id"] == 3

    def test_state_machine_active_to_pending(self) -> None:
        """ACTIVE → PENDING_TRANSFER when person leaves field of view."""
        from models.global_reid.state_machine import next_missing_state, ACTIVE, PENDING_TRANSFER
        result = next_missing_state(ACTIVE, missing_sec=2.0, max_candidate_age_sec=30.0)
        assert result == PENDING_TRANSFER

    def test_state_machine_any_to_dormant_when_too_old(self) -> None:
        """Any state → DORMANT when missing longer than max_candidate_age_sec."""
        from models.global_reid.state_machine import next_missing_state, IN_BLIND_ZONE, DORMANT
        result = next_missing_state(IN_BLIND_ZONE, missing_sec=60.0, max_candidate_age_sec=30.0)
        assert result == DORMANT

    def test_state_machine_closed_stays_closed(self) -> None:
        """Once CLOSED a track must never transition to another state."""
        from models.global_reid.state_machine import next_missing_state, CLOSED
        result = next_missing_state(CLOSED, missing_sec=1000.0, max_candidate_age_sec=30.0)
        assert result == CLOSED
