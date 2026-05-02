"""Tracking JSON schema dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.common.schemas import Track


@dataclass(slots=True)
class TrackFrameRecord:
    frame_id: int
    timestamp_sec: float
    camera_id: str
    tracks: list[Track] = field(default_factory=list)
    failure_reason: str = "OK"


__all__ = ["Track", "TrackFrameRecord"]
