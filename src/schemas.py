from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FrameRecord:
    frame_id: int
    timestamp_sec: float
    camera_id: str
    failure_reason: str = "OK"


@dataclass(slots=True)
class Detection:
    bbox: list[float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    failure_reason: str = "OK"


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: list[float]
    confidence: float
    class_name: str = "person"
    age: int = 0
    hits: int = 0
    misses: int = 0
    is_confirmed: bool = False
    fragment_count: int = 0
    quality_score: float = 0.0
    failure_reason: str = "OK"


@dataclass(slots=True)
class Keypoint:
    id: int
    name: str
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class PosePerson:
    track_id: int | None
    bbox: list[float]
    keypoints: list[Keypoint]
    is_confirmed: bool
    visible_keypoint_count: int
    visible_keypoint_ratio: float
    pose_track_iou: float | None
    failure_reason: str = "OK"


@dataclass(slots=True)
class ADLEvent:
    frame_id: int
    timestamp_sec: float
    camera_id: str
    track_id: int
    raw_label: str
    smoothed_label: str
    adl_label: str
    confidence: float | None
    window_size: int
    visible_keypoint_ratio: float | None
    failure_reason: str = "OK"


@dataclass(slots=True)
class FaceEvent:
    frame_id: int
    timestamp_sec: float
    camera_id: str
    track_id: int
    face_detected: bool
    face_bbox: list[float] | None
    embedding_dim: int | None
    embedding: list[float] | None
    face_quality: float | None
    spoof_status: str
    failure_reason: str = "OK"


@dataclass(slots=True)
class ReIDPersonRecord:
    local_track_id: int
    global_id: str
    state: str
    match_status: str
    bbox: list[float]
    adl_label: str | None
    score_total: float | None
    score_face: float | None
    score_body: float | None
    score_pose: float | None
    score_height: float | None
    score_time: float | None
    score_topology: float | None
    topology_allowed: bool | None
    delta_time_sec: float | None
    entry_zone: str | None
    exit_zone: str | None
    failure_reason: str = "OK"


@dataclass(slots=True)
class ModuleMetrics:
    metric_type: str
    total_frames: int | None
    processed_frames: int | None
    elapsed_sec: float | None
    fps_processing: float | None
    avg_latency_ms_per_frame: float | None
    model_info: dict[str, Any]
    input_video: str | None
    camera_id: str | None
    start_time: str | None
    output_paths: dict[str, Any]
    failure_reason: str = "OK"


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
