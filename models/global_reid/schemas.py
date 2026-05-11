"""Global ReID JSON schema dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.schemas import ReIDPersonRecord


@dataclass(slots=True)
class ReIDFrameRecord:
    frame_id: int
    timestamp_sec: float
    camera_id: str
    persons: list[ReIDPersonRecord] = field(default_factory=list)
    failure_reason: str = "OK"


@dataclass(slots=True)
class GlobalPersonTableEntry:
    global_id: str
    state: str
    last_camera_id: str
    last_seen_sec: float | None
    total_sightings: int
    failure_reason: str = "OK"
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["GlobalPersonTableEntry", "ReIDFrameRecord", "ReIDPersonRecord"]
