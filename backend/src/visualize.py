"""
Visualization utilities - pose skeleton & UI
"""
import cv2
import numpy as np

# Keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Skeleton connections with color indices
# Each connection has: (start_kp, end_kp, color_name)
SKELETON_LIMBS = [
    # Face - Cyan
    (KP_NOSE, KP_LEFT_EYE, "face"),
    (KP_NOSE, KP_RIGHT_EYE, "face"),
    (KP_LEFT_EYE, KP_LEFT_EAR, "face"),
    (KP_RIGHT_EYE, KP_RIGHT_EAR, "face"),
    # Torso - Yellow
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, "torso"),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP, "torso"),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP, "torso"),
    (KP_LEFT_HIP, KP_RIGHT_HIP, "torso"),
    # Left arm - Green
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, "left_arm"),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST, "left_arm"),
    # Right arm - Blue
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, "right_arm"),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST, "right_arm"),
    # Left leg - Magenta
    (KP_LEFT_HIP, KP_LEFT_KNEE, "left_leg"),
    (KP_LEFT_KNEE, KP_LEFT_ANKLE, "left_leg"),
    # Right leg - Orange
    (KP_RIGHT_HIP, KP_RIGHT_KNEE, "right_leg"),
    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE, "right_leg"),
]

# Limb colors (BGR) - Colorful skeleton
LIMB_COLORS = {
    "face": (255, 255, 0),      # Cyan
    "torso": (0, 255, 255),     # Yellow
    "left_arm": (0, 255, 0),    # Green
    "right_arm": (255, 0, 0),   # Blue
    "left_leg": (255, 0, 255),  # Magenta
    "right_leg": (0, 165, 255), # Orange
}

# Keypoint colors by body part
KEYPOINT_COLORS = {
    0: (255, 255, 0),   # Nose - Cyan
    1: (255, 255, 0),   # Left eye
    2: (255, 255, 0),   # Right eye
    3: (255, 255, 0),   # Left ear
    4: (255, 255, 0),   # Right ear
    5: (0, 255, 0),     # Left shoulder - Green
    6: (255, 0, 0),     # Right shoulder - Blue
    7: (0, 255, 0),     # Left elbow
    8: (255, 0, 0),     # Right elbow
    9: (0, 255, 0),     # Left wrist
    10: (255, 0, 0),    # Right wrist
    11: (255, 0, 255),  # Left hip - Magenta
    12: (0, 165, 255),  # Right hip - Orange
    13: (255, 0, 255),  # Left knee
    14: (0, 165, 255),  # Right knee
    15: (255, 0, 255),  # Left ankle
    16: (0, 165, 255),  # Right ankle
}

# Legacy for reference
SKELETON_CONNECTIONS = [
    (KP_NOSE, KP_LEFT_EYE), (KP_NOSE, KP_RIGHT_EYE),
    (KP_LEFT_EYE, KP_LEFT_EAR), (KP_RIGHT_EYE, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP), (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW), (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST), (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_HIP, KP_LEFT_KNEE), (KP_RIGHT_HIP, KP_RIGHT_KNEE),
    (KP_LEFT_KNEE, KP_LEFT_ANKLE), (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
]

# Colors - Postures (FALL_DOWN is RED for danger)
POSTURE_COLORS = {
    "WALKING": (255, 255, 0),     # Cyan
    "SITTING": (0, 165, 255),     # Orange
    "FALL_DOWN": (0, 0, 255),     # Red - DANGER
    "": (100, 100, 100),          # Gray - default/no label
}

# Unknown/Unmatched color
UNMATCHED_COLOR = (0, 0, 255)  # RED for unmatched tracks

GLOBAL_ID_COLORS = {
    1: (0, 255, 0),        # Bright Green
    2: (255, 0, 0),        # Bright Blue
    3: (0, 165, 255),      # Orange
    4: (255, 0, 255),      # Magenta
    5: (255, 255, 0),      # Cyan
    6: (0, 255, 255),      # Yellow
    7: (147, 20, 255),     # Deep Pink
    8: (203, 192, 255),    # Pink
    9: (128, 0, 128),      # Purple
    10: (0, 128, 255),     # Dark Orange
}


def draw_skeleton(frame, keypoints, bbox_color=None, min_conf=0.3, colorful=True):
    """Draw pose skeleton with colorful limbs."""
    if keypoints is None or len(keypoints) < 17:
        return
    
    kpts = np.array(keypoints)
    
    if colorful:
        # Draw colorful limbs
        for start_idx, end_idx, limb_name in SKELETON_LIMBS:
            if kpts[start_idx][2] > min_conf and kpts[end_idx][2] > min_conf:
                pt1 = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                pt2 = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                color = LIMB_COLORS[limb_name]
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw colorful keypoints
        for i, kp in enumerate(kpts):
            if kp[2] > min_conf:
                pt = (int(kp[0]), int(kp[1]))
                color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                cv2.circle(frame, pt, 5, color, -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 1)
    else:
        # Single color mode (use bbox_color)
        color = bbox_color if bbox_color else (0, 255, 0)
        for start_idx, end_idx, _ in SKELETON_LIMBS:
            if kpts[start_idx][2] > min_conf and kpts[end_idx][2] > min_conf:
                pt1 = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                pt2 = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                cv2.line(frame, pt1, pt2, color, 2)
        
        for kp in kpts:
            if kp[2] > min_conf:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, pt, 4, color, -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)


def get_color_for_id(gid):
    """Get color for global ID."""
    return GLOBAL_ID_COLORS.get(gid, (100, 100, 100))


def draw_ui_panel(frame, cam_id, frame_idx, total_frames, is_master, global_ids, person_profiles=None):
    """Draw UI panel."""
    h, w = frame.shape[:2]
    
    if person_profiles is None:
        person_profiles = {}
    
    # Top panel
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    
    mode = "REGISTRATION" if is_master else "SLAVE - Match Only"
    mode_color = (0, 255, 255) if is_master else (255, 150, 0)
    
    cv2.putText(frame, cam_id.upper(), (15, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)
    cv2.putText(frame, mode, (130, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (15, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "POSE + ADL + ReID", (w - 200, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Global IDs legend (with profile info)
    if global_ids:
        legend_x = w - 280
        legend_y = 110
        cv2.rectangle(frame, (legend_x - 10, legend_y - 20),
                    (w - 10, legend_y + len(global_ids) * 30 + 10),
                    (30, 30, 30), -1)
        cv2.putText(frame, "Global IDs:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, gid in enumerate(global_ids):
            color = get_color_for_id(gid)
            y_pos = legend_y + 25 + i * 30
            cv2.circle(frame, (legend_x + 10, y_pos - 5), 8, color, -1)
            
            # Show profile info if available
            if gid in person_profiles:
                p = person_profiles[gid]
                gender_short = "M" if p.get('gender') == 'male' else "F"
                legend_text = f"G{gid} - {p['name']} ({p.get('age')}, {gender_short})"
            else:
                legend_text = f"Person {gid}"
            
            cv2.putText(frame, legend_text,
                       (legend_x + 25, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Progress bar
    bar_y = h - 25
    progress = frame_idx / total_frames if total_frames > 0 else 0
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, bar_y),
                 (10 + int((w - 20) * progress), bar_y + 15),
                 mode_color, -1)
    
    cv2.putText(frame, "SPACE=Pause | N=Next | Q=Quit",
               (w - 300, h - 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

