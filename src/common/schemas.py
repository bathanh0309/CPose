from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class Detection:
    bbox: list[float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: list[float]
    confidence: float
    class_name: str = "person"


@dataclass(slots=True)
class Keypoint:
    id: int
    name: str
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class ADLEvent:
    frame_id: int
    timestamp_sec: float
    track_id: int
    adl_label: str
    confidence: float
    window_size: int


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value

