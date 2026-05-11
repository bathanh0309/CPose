from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    OK = "OK"
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"
    TRACK_FRAGMENTED = "TRACK_FRAGMENTED"
    UNCONFIRMED_TRACK = "UNCONFIRMED_TRACK"
    SHORT_TRACK_WINDOW = "SHORT_TRACK_WINDOW"
    LOW_KEYPOINT_VISIBILITY = "LOW_KEYPOINT_VISIBILITY"
    NO_FACE = "NO_FACE"
    BODY_OCCLUDED = "BODY_OCCLUDED"
    TOPOLOGY_CONFLICT = "TOPOLOGY_CONFLICT"
    TIME_WINDOW_CONFLICT = "TIME_WINDOW_CONFLICT"
    MULTI_CANDIDATE_CONFLICT = "MULTI_CANDIDATE_CONFLICT"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_VIDEO = "INVALID_VIDEO"
    INVALID_MANIFEST = "INVALID_MANIFEST"
    INVALID_TOPOLOGY = "INVALID_TOPOLOGY"
    MISSING_INPUT_JSON = "MISSING_INPUT_JSON"
    GROUND_TRUTH_NOT_FOUND = "GROUND_TRUTH_NOT_FOUND"
    STEP_FAILED = "STEP_FAILED"


def as_reason(value: ErrorCode | str | None) -> str:
    if value is None:
        return ErrorCode.STEP_FAILED.value
    if isinstance(value, ErrorCode):
        return value.value
    try:
        return ErrorCode(str(value)).value
    except ValueError:
        return ErrorCode.STEP_FAILED.value


__all__ = ["ErrorCode", "as_reason"]
