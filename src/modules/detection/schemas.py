"""Detection JSON schema dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.common.schemas import Detection


@dataclass(slots=True)
class DetectionFrameRecord:
    frame_id: int
    timestamp_sec: float
    camera_id: str
    detections: list[Detection] = field(default_factory=list)
    failure_reason: str = "OK"


__all__ = ["Detection", "DetectionFrameRecord"]
