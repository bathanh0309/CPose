from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(slots=True)
class BBox:
    """2D bounding box in x1, y1, x2, y2 format."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(slots=True)
class Pose2D:
    """Skeleton keypoints and optional confidence scores."""

    keypoints: np.ndarray
    scores: Optional[np.ndarray] = None


@dataclass(slots=True)
class ActionState:
    label: Optional[str] = None
    score: float = 0.0
    model_name: Optional[str] = None


@dataclass(slots=True)
class Track:
    """Tracked person state used by the Flask app recognizer pipeline."""

    local_id: int
    global_id: Optional[int] = None
    cam_id: str = ""
    timestamp: float = 0.0
    bbox: Optional[BBox] = None
    pose: Optional[Pose2D] = None
    bbox_score: float = 0.0
    track_confidence: float = 0.0
    action: ActionState = field(default_factory=ActionState)
    reid_embedding: Optional[np.ndarray] = None
    reid_score: float = 0.0
    identity_name: Optional[str] = None
    identity_id: Optional[str] = None
    face_similarity: float = 0.0
    has_face: bool = False
    is_active: bool = True
    is_occluded: bool = False
    lost_frames: int = 0
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class GlobalTrack:
    global_id: int
    identity_name: Optional[str] = None
    identity_id: Optional[str] = None
    history: list[Track] = field(default_factory=list)
    body_feature_bank: list[np.ndarray] = field(default_factory=list)
    face_feature_bank: list[np.ndarray] = field(default_factory=list)

    def add_observation(self, track: Track) -> None:
        self.history.append(track)
        if track.reid_embedding is not None:
            self.body_feature_bank.append(track.reid_embedding)
        if track.has_face and track.identity_id is not None:
            face_embedding = track.meta.get("face_embedding")
            if face_embedding is not None:
                self.face_feature_bank.append(face_embedding)

