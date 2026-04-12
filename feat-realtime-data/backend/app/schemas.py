# Schema Pydantic dùng chung cho API, WebSocket, camera và event.
"""
Shared Pydantic schemas used across API routes and services.
All timestamps are stored/returned as ISO 8601 strings (UTC).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class StreamFormat(str, Enum):
    MJPEG = "MJPEG"
    MPEG = "MPEG"
    HLS = "HLS"
    WebRTC = "WebRTC"


class EventType(str, Enum):
    person_detected = "person_detected"
    person_left = "person_left"
    recording_started = "recording_started"
    recording_stopped = "recording_stopped"
    camera_online = "camera_online"
    camera_offline = "camera_offline"


# ── Camera schemas ─────────────────────────────────────────────────────────

class CameraOut(BaseModel):
    id: str                        # e.g. "cam01"
    name: str                      # e.g. "Cam 01"
    masked_rtsp_source: str
    rtsp_configured: bool
    online: bool
    source_fps: float
    detect_fps: float
    preview_fps: float
    decoder_errors: int
    preview_enabled: bool
    preview_url: str               # MJPEG endpoint path
    stream_format: StreamFormat
    last_seen_at: Optional[str] = None  # ISO timestamp


class CameraStatusUpdate(BaseModel):
    """Pushed over WebSocket when camera state changes."""
    id: str
    online: Optional[bool] = None
    source_fps: Optional[float] = None
    detect_fps: Optional[float] = None
    preview_fps: Optional[float] = None
    decoder_errors: Optional[int] = None
    last_seen_at: Optional[str] = None
    preview_enabled: Optional[bool] = None


# ── Event schemas ──────────────────────────────────────────────────────────

class EventOut(BaseModel):
    id: str
    time: str                   # full ISO timestamp (stored internally)
    event: EventType
    person: str = "Unknown"    # no identity recognition — always Unknown
    cam: str                   # display name, e.g. "Cam 02"
    camera_id: str             # internal id, e.g. "cam02"
    clip_path: Optional[str] = None


class EventCreate(BaseModel):
    """Internal model used by services to emit events."""
    event: EventType
    camera_id: str
    cam_name: str
    person: str = "Unknown"
    clip_path: Optional[str] = None
    time: datetime = Field(default_factory=datetime.utcnow)


# ── WebSocket message schemas ───────────────────────────────────────────────

class WsEventMessage(BaseModel):
    type: Literal["event"] = "event"
    payload: EventOut


class WsCameraStatusMessage(BaseModel):
    type: Literal["camera_status"] = "camera_status"
    payload: CameraStatusUpdate


class WsPingMessage(BaseModel):
    type: Literal["ping"] = "ping"
    payload: dict = Field(default_factory=dict)


# ── Health ─────────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str = "ok"
    uptime: float               # seconds since startup
    cameras_online: int
    cameras_total: int
    ws_clients: int


class ServiceState(str, Enum):
    active = "active"
    stopped = "stopped"


class ProcessingJobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ServiceOut(BaseModel):
    status: ServiceState
    cameras_total: int
    cameras_online: int


class ConfigCameraOut(BaseModel):
    id: str
    name: str
    masked_rtsp_source: str


class ResourcesUploadOut(BaseModel):
    file_name: str
    skipped_lines: int
    cameras: list[ConfigCameraOut]
    service_status: ServiceState
    uploaded_at: str


class ConfigStatusOut(BaseModel):
    config_loaded: bool
    file_name: Optional[str] = None
    uploaded_at: Optional[str] = None
    cameras_configured: int = 0


class ProcessingJobOut(BaseModel):
    raw_file_name: str
    raw_path: str
    raw_size_bytes: int
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    status: ProcessingJobStatus
    processed_file_name: Optional[str] = None
    processed_path: Optional[str] = None
    processed_url: Optional[str] = None
    processed_size_bytes: Optional[int] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ProcessingQueueOut(BaseModel):
    raw_videos_dir: str
    processed_videos_dir: str
    queued: int
    processing: int
    completed: int
    failed: int
    jobs: list[ProcessingJobOut]
