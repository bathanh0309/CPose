from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ADL_CLASSES = [
    "standing",
    "sitting",
    "walking",
    "lying_down",
    "falling",
    "reaching",
    "bending",
    "unknown",
]


@dataclass(slots=True)
class ADLConfig:
    window_size: int = 30
    min_visible_keypoints: int = 8
    min_visible_ratio: float = 0.50
    smoothing_frames: int = 7
    knee_sitting_angle: float = 125.0
    walking_velocity: float = 0.05
    falling_velocity: float = 0.08
    lying_aspect_ratio: float = 1.15
    torso_horizontal_low: float = 35.0
    torso_lean_low: float = 35.0
    torso_lean_high: float = 62.0


def adl_config_from_dict(payload: dict[str, Any] | None = None, **overrides: Any) -> ADLConfig:
    data = dict(payload or {})
    data.update({k: v for k, v in overrides.items() if v is not None})
    fields = ADLConfig.__dataclass_fields__
    return ADLConfig(**{key: data[key] for key in fields if key in data})
