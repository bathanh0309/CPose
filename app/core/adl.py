import math
import time
import numpy as np
from collections import deque, Counter
from typing import Optional

from app.utils.pose_utils import rule_based_adl
from app.utils.runtime_config import get_runtime_section

_ADL_CFG = get_runtime_section("adl")


class ADLConfig:
    """ADL runtime config.

    Thresholds that govern real classification live entirely in
    `pose_utils.py` / `configs/unified_config.yaml [pose_utils]` section.
    The constants below are kept only for legacy TrackState helpers
    (movement detection, posture voting UI), NOT for ADL inference.
    """

    # Movement / UI helpers — NOT used by rule_based_adl()
    MOVEMENT_THRESHOLD_RATIO = float(_ADL_CFG.get("movement_threshold_ratio", 0.025))
    MOVEMENT_WALKING_MULTIPLIER = float(_ADL_CFG.get("movement_walking_multiplier", 1.2))
    KEYPOINT_CONF = float(_ADL_CFG.get("keypoint_conf", 0.25))
    HAND_RAISE_FRAMES = int(_ADL_CFG.get("hand_raise_frames", 10))
    POSTURE_VOTING_FRAMES = int(_ADL_CFG.get("posture_voting_frames", 3))

    @classmethod
    def from_dict(cls, adl_config: dict):
        """Load from config dict (only non-classifier fields)."""
        if adl_config:
            cls.MOVEMENT_THRESHOLD_RATIO = adl_config.get("movement_threshold_ratio", cls.MOVEMENT_THRESHOLD_RATIO)
            cls.MOVEMENT_WALKING_MULTIPLIER = adl_config.get("movement_walking_multiplier", cls.MOVEMENT_WALKING_MULTIPLIER)
            cls.KEYPOINT_CONF = adl_config.get("keypoint_conf", cls.KEYPOINT_CONF)
            cls.HAND_RAISE_FRAMES = adl_config.get("hand_raise_frames", cls.HAND_RAISE_FRAMES)
            cls.POSTURE_VOTING_FRAMES = adl_config.get("posture_voting_frames", cls.POSTURE_VOTING_FRAMES)


# Tracking
POSITION_HISTORY_MAXLEN = int(_ADL_CFG.get("position_history_maxlen", 30))
POSTURE_HISTORY_MAXLEN = int(_ADL_CFG.get("posture_history_maxlen", 10))
EVENT_HISTORY_MAXLEN = int(_ADL_CFG.get("event_history_maxlen", 5))
DEFAULT_KNEE_ANGLE = float(_ADL_CFG.get("default_knee_angle", 180))
ASSUMED_FPS = float(_ADL_CFG.get("assumed_fps", 30))

# Keypoint indices (COCO-17)
KP_NOSE = 0
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Canonical ADL labels (CLAUDE.md §17.3 — all lowercase).
# Used ONLY when an external caller needs a non-standard display string.
# Phase 3 pipeline writes the lowercase labels directly; this map is for
# legacy compatibility with callers that expect uppercase variants.
_LEGACY_LABEL_MAP: dict[str, str] = {
    "standing": "standing",
    "sitting": "sitting",
    "walking": "walking",
    "lying_down": "lying_down",
    "falling": "falling",
    "reaching": "reaching",
    "bending": "bending",
    "unknown": "",
}


class TrackState:
    """Track state with ADL features (used by realtime/live pipeline helpers)."""

    def __init__(self, track_id, frame_height):
        self.track_id = track_id
        self.frame_height = frame_height
        self.global_id = None

        # Movement
        self.positions = deque(maxlen=POSITION_HISTORY_MAXLEN)
        self.last_center = None

        # Posture
        self.postures = deque(maxlen=POSTURE_HISTORY_MAXLEN)
        self.current_posture = ""
        self.prev_posture = ""

        # Events
        self.events = deque(maxlen=EVENT_HISTORY_MAXLEN)

    def update_position(self, bbox):
        """Update position."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.last_center = (cx, cy)
        self.positions.append((time.time(), cx, cy))

    def get_movement_in_window(self, window_sec=1.0):
        """Get movement distance."""
        if len(self.positions) < 2:
            return 0.0

        now = time.time()
        total_dist = 0.0
        prev = None

        for t, x, y in self.positions:
            if now - t > window_sec:
                continue
            if prev is not None:
                dx = x - prev[0]
                dy = y - prev[1]
                total_dist += math.sqrt(dx * dx + dy * dy)
            prev = (x, y)

        return total_dist

    def add_posture(self, posture):
        """Add posture with voting."""
        self.postures.append(posture)

        if len(self.postures) >= ADLConfig.POSTURE_VOTING_FRAMES:
            counter = Counter(self.postures)
            voted = counter.most_common(1)[0][0]

            if voted != self.current_posture:
                self.prev_posture = self.current_posture
                self.current_posture = voted

                # Detect fall event
                if voted == "falling" and self.prev_posture not in ["falling", ""]:
                    self.add_event("FALL DETECTED")
        else:
            self.current_posture = posture

    def add_event(self, event):
        """Add event."""
        timestamp = time.strftime("%H:%M:%S")
        self.events.append(f"{timestamp} - {event}")

    def check_hand_raise(self, keypoints):
        """Disabled - return None."""
        return None


def classify_posture(keypoints, bbox, track_state, frame_height):
    """Compatibility wrapper over the shared rule-based ADL classifier.

    Delegates to `pose_utils.rule_based_adl()` — the single canonical
    classifier. Returns a canonical lowercase label string.
    """
    if keypoints is None:
        return ""
    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[0] < 17 or keypoints.shape[1] < 3:
        return ""

    window = [(keypoints[:, :2], keypoints[:, 2])]
    label, _ = rule_based_adl(window, {})
    # Return canonical lowercase map value (same as label for most cases)
    return _LEGACY_LABEL_MAP.get(label, label)
