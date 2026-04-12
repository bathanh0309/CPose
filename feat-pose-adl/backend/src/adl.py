"""
ADL (Activities of Daily Living) Classification
Posture detection + Event detection
"""
import math
import time
import numpy as np
from collections import deque, Counter
from typing import Optional


class ADLConfig:
    """ADL configuration - can be loaded from config dict."""
    
    # Default values
    TORSO_ANGLE_LAYING = 35
    ASPECT_RATIO_LAYING = 0.9
    KNEE_ANGLE_SITTING = 140
    MOVEMENT_THRESHOLD_RATIO = 0.025
    MOVEMENT_WALKING_MULTIPLIER = 1.2
    KEYPOINT_CONF = 0.25
    HAND_RAISE_FRAMES = 10
    POSTURE_VOTING_FRAMES = 3
    
    @classmethod
    def from_dict(cls, adl_config: dict):
        """Load from config dict."""
        if adl_config:
            cls.TORSO_ANGLE_LAYING = adl_config.get('torso_angle_laying', 35)
            cls.ASPECT_RATIO_LAYING = adl_config.get('aspect_ratio_laying', 0.9)
            cls.KNEE_ANGLE_SITTING = adl_config.get('knee_angle_sitting', 140)
            cls.MOVEMENT_THRESHOLD_RATIO = adl_config.get('movement_threshold_ratio', 0.025)
            cls.MOVEMENT_WALKING_MULTIPLIER = adl_config.get('movement_walking_multiplier', 1.2)
            cls.KEYPOINT_CONF = adl_config.get('keypoint_conf', 0.25)
            cls.HAND_RAISE_FRAMES = adl_config.get('hand_raise_frames', 10)
            cls.POSTURE_VOTING_FRAMES = adl_config.get('posture_voting_frames', 3)


# Tracking
POSITION_HISTORY_MAXLEN = 30
POSTURE_HISTORY_MAXLEN = 10
EVENT_HISTORY_MAXLEN = 5
DEFAULT_KNEE_ANGLE = 180
ASSUMED_FPS = 30

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


class TrackState:
    """Track state with ADL features."""
    
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
                total_dist += math.sqrt(dx*dx + dy*dy)
            prev = (x, y)
        
        return total_dist
    
    def add_posture(self, posture):
        """Add posture with voting - simplified."""
        self.postures.append(posture)
        
        if len(self.postures) >= ADLConfig.POSTURE_VOTING_FRAMES:
            counter = Counter(self.postures)
            voted = counter.most_common(1)[0][0]
            
            if voted != self.current_posture:
                self.prev_posture = self.current_posture
                self.current_posture = voted
                
                # Only detect FALL_DOWN event
                if voted == "FALL_DOWN" and self.prev_posture not in ["FALL_DOWN", ""]:
                    self.add_event(" FALL DETECTED")
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
    """Classify posture from keypoints.
    
    IMPORTANT: Only classify FALL_DOWN if enough keypoints are visible.
    If only head/shoulders visible (high angle camera), don't assume fall.
    """
    if keypoints is None or len(keypoints) < 17:
        return ""
    
    def get_point(idx):
        if keypoints[idx][2] > ADLConfig.KEYPOINT_CONF:
            return keypoints[idx][:2]
        return None
    
    # Get key body points
    l_sh = get_point(KP_LEFT_SHOULDER)
    r_sh = get_point(KP_RIGHT_SHOULDER)
    l_hip = get_point(KP_LEFT_HIP)
    r_hip = get_point(KP_RIGHT_HIP)
    l_knee = get_point(KP_LEFT_KNEE)
    r_knee = get_point(KP_RIGHT_KNEE)
    l_ankle = get_point(KP_LEFT_ANKLE)
    r_ankle = get_point(KP_RIGHT_ANKLE)
    
    # Count visible keypoints
    visible_keypoints = sum(1 for kp in keypoints if kp[2] > ADLConfig.KEYPOINT_CONF)
    has_shoulders = l_sh is not None or r_sh is not None
    has_hips = l_hip is not None or r_hip is not None
    has_knees = l_knee is not None or r_knee is not None
    has_lower_body = has_knees or l_ankle is not None or r_ankle is not None
    
    # Minimum requirements for different classifications
    can_classify_fall = has_shoulders and has_hips  # Need upper body
    can_classify_sitting = has_hips and has_knees   # Need hips + knees
    
    # Torso angle (only calculate if we have enough points)
    torso_angle = 0
    if has_shoulders and has_hips:
        mid_sh = (l_sh + r_sh) / 2 if l_sh is not None and r_sh is not None else (l_sh if l_sh is not None else r_sh)
        mid_hip = (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None else (l_hip if l_hip is not None else r_hip)
        dx = abs(mid_sh[0] - mid_hip[0])
        dy = abs(mid_sh[1] - mid_hip[1])
        torso_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0
    
    # Knee angle for sitting detection
    knee_angles = []
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        v1 = l_hip - l_knee
        v2 = l_ankle - l_knee
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        knee_angles.append(math.degrees(math.acos(cos_angle)))
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        v1 = r_hip - r_knee
        v2 = r_ankle - r_knee
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        knee_angles.append(math.degrees(math.acos(cos_angle)))
    avg_knee_angle = np.mean(knee_angles) if knee_angles else DEFAULT_KNEE_ANGLE
    
    # Movement & aspect ratio
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    aspect_ratio = w / h if h > 0 else 0
    movement = track_state.get_movement_in_window(1.0)
    movement_threshold = frame_height * ADLConfig.MOVEMENT_THRESHOLD_RATIO
    
    # ========== CLASSIFY ==========
    
    # 1. FALL_DOWN: Only if we can see shoulders + hips
    # 1. FALL_DOWN: Strict check
    # Must see lower body to confirm fall (avoid false positive when bowing/zoomed in)
    if can_classify_fall and has_lower_body:
        is_torso_horizontal = torso_angle > (ADLConfig.TORSO_ANGLE_LAYING + 10) # Stricter angle
        is_wide_with_body = aspect_ratio > ADLConfig.ASPECT_RATIO_LAYING
        
        if is_torso_horizontal or is_wide_with_body:
            return "FALL_DOWN"
    
    # 2. SITTING
    if can_classify_sitting and avg_knee_angle < ADLConfig.KNEE_ANGLE_SITTING:
        if movement < movement_threshold:
            return "SITTING"
    
    # 3. WALKING: Moving AND valid lower body detection
    if movement > movement_threshold * ADLConfig.MOVEMENT_WALKING_MULTIPLIER:
        if has_lower_body: # Only say walking if feet/knees are visible
            return "WALKING"
    
    # Default: No label (not enough info or standing still)
    return ""

