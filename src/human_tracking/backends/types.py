from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class BBox:
    """2D Bounding Box (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class Pose2D:
    """Skeleton data: keypoints and optional scores."""
    keypoints: np.ndarray          # shape (J, 2) or (J, 3)
    scores: Optional[np.ndarray] = None  # (J,)

@dataclass
class ActionState:
    """Action/Pose classification result."""
    label: Optional[str] = None    # "standing", "sitting", "walking", "lying_down", "falling"
    score: float = 0.0
    model_name: Optional[str] = None

@dataclass
class Track:
    """
    Core object representing a person tracked within a single camera frame.
    Connects Detection, Pose, Tracking, ADL, ReID, and FaceID.
    """
    # Identification
    local_id: int                    # ID from single-cam tracker (e.g. DeepSort)
    global_id: Optional[int] = None  # Unified ID across cameras (from MultiCamAssociator)

    # Spatio-temporal info
    cam_id: str = ""
    timestamp: float = 0.0
    bbox: Optional[BBox] = None
    pose: Optional[Pose2D] = None
    bbox_score: float = 0.0
    track_confidence: float = 0.0

    # ADL / Action
    action: ActionState = field(default_factory=ActionState)

    # Body ReID
    reid_embedding: Optional[np.ndarray] = None  # Feature vector from Body ReID model
    reid_score: float = 0.0                      # Match confidence in cross-camera association

    # Face ID
    identity_name: Optional[str] = None          # Registered name from FaceRegistry
    identity_id: Optional[str] = None            # Person ID in FaceRegistry (may map to global_id)
    face_similarity: float = 0.0                 # Cosine similarity with best registry match
    has_face: bool = False                       # Whether a valid face was cropped in this frame

    # Status
    is_active: bool = True                       # Present in current frame
    is_occluded: bool = False
    lost_frames: int = 0                         # Frames missing since last successful detection

    # Metadata (for research logs/ancillary data)
    meta: Dict = field(default_factory=dict)

@dataclass
class GlobalTrack:
    """
    Aggregates person snapshots across time and multiple cameras.
    Maintains a trajectory and feature bank for a unique global identity.
    """
    global_id: int

    identity_name: Optional[str] = None
    identity_id: Optional[str] = None

    # Temporal history of observations (List of Track snapshots)
    history: List[Track] = field(default_factory=list)

    # Aggregated feature banks for stability
    body_feature_bank: List[np.ndarray] = field(default_factory=list)
    face_feature_bank: List[np.ndarray] = field(default_factory=list)

    def add_observation(self, track: Track):
        """Append a new per-frame observation to the global trajectory."""
        self.history.append(track)
        if track.reid_embedding is not None:
            self.body_feature_bank.append(track.reid_embedding)
        if track.has_face and track.identity_id is not None:
            face_emb = track.meta.get("face_embedding")
            if face_emb is not None:
                self.face_feature_bank.append(face_emb)
