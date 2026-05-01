from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    OK = "OK"
    MODEL_MISSING = "MODEL_MISSING"
    CONFIG_MISSING = "CONFIG_MISSING"
    VIDEO_OPEN_FAILED = "VIDEO_OPEN_FAILED"
    UNCONFIRMED_TRACK = "UNCONFIRMED_TRACK"
    NO_MATCHED_TRACK = "NO_MATCHED_TRACK"
    STEP_FAILED = "STEP_FAILED"
