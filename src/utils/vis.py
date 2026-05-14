"""
src/utils/vis.py — CPose visualization utilities
"""
import time

import cv2
import numpy as np

# ──────────────────────────────────────────────
# COCO-17 skeleton edges (joint_a, joint_b)
# ──────────────────────────────────────────────
COCO_EDGES = [
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 6),                  # shoulders
    (5, 11), (6, 12),        # torso sides
    (11, 12),                # hips
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
    (0, 5), (0, 6),          # neck → shoulders (approx)
]

COCO_JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Màu mỗi edge: (B, G, R)
EDGE_COLORS = [
    (255, 128, 0),   # left arm
    (255, 128, 0),
    (0, 128, 255),   # right arm
    (0, 128, 255),
    (0, 255, 128),   # shoulders
    (200, 200, 0),   # torso
    (200, 200, 0),
    (200, 200, 0),
    (0, 200, 200),   # left leg
    (0, 200, 200),
    (180, 0, 255),   # right leg
    (180, 0, 255),
    (100, 200, 100), # neck
    (100, 200, 100),
]

# Màu track_id (30 màu vòng)
_PALETTE = [
    (0, 220, 0), (0, 120, 255), (255, 80, 0), (255, 0, 180),
    (0, 220, 220), (180, 0, 255), (255, 200, 0), (0, 255, 128),
    (255, 100, 100), (100, 100, 255), (0, 180, 180), (200, 200, 0),
    (255, 0, 80), (80, 255, 0), (0, 80, 255), (255, 160, 0),
    (160, 0, 255), (0, 255, 200), (255, 60, 120), (60, 255, 120),
    (120, 60, 255), (200, 100, 0), (0, 200, 100), (100, 0, 200),
    (255, 200, 100), (100, 255, 200), (200, 100, 255), (80, 80, 0),
    (0, 80, 80), (80, 0, 80),
]


def track_color(track_id: int):
    """Trả màu BGR ổn định cho track_id."""
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ──────────────────────────────────────────────
# Core draw functions
# ──────────────────────────────────────────────

def draw_detection(frame, det: dict, label: str = None, color=None):
    """Vẽ bbox + keypoint skeleton lên frame (in-place).

    Args:
        frame:  BGR numpy array, sẽ bị modify in-place.
        det:    Detection dict theo contract CLAUDE.md §3.
        label:  Text hiển thị phía trên bbox.
        color:  (B,G,R). Nếu None, chọn theo track_id.
    """
    tid = det.get("track_id", -1)
    if color is None:
        color = track_color(tid) if tid >= 0 else (0, 165, 255)

    x1, y1, x2, y2 = map(int, det["bbox"])

    # Bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = x1, max(th + 4, y1 - 2)
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(
            frame, label, (lx + 2, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    kpts = det.get("keypoints")
    kpt_scores = det.get("keypoint_scores")

    if kpts is not None:
        # Skeleton edges
        for idx, (a, b) in enumerate(COCO_EDGES):
            if a < len(kpts) and b < len(kpts):
                ca = 1.0 if kpt_scores is None else float(kpt_scores[a])
                cb = 1.0 if kpt_scores is None else float(kpt_scores[b])
                if ca > 0.2 and cb > 0.2:
                    xa, ya = int(kpts[a][0]), int(kpts[a][1])
                    xb, yb = int(kpts[b][0]), int(kpts[b][1])
                    ec = EDGE_COLORS[idx] if idx < len(EDGE_COLORS) else (255, 180, 0)
                    cv2.line(frame, (xa, ya), (xb, yb), ec, 2, cv2.LINE_AA)

        # Joints
        for i, kp in enumerate(kpts):
            conf = 1.0 if kpt_scores is None else float(kpt_scores[i])
            if conf > 0.2:
                cx, cy = int(kp[0]), int(kp[1])
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 3, color, -1)

    return frame


def draw_skeleton_only(frame, keypoints, keypoint_scores=None, color=(0, 220, 0)):
    """Vẽ skeleton thuần (không bbox) — dùng cho ADL visualization."""
    kpts = keypoints
    kpt_scores = keypoint_scores

    for idx, (a, b) in enumerate(COCO_EDGES):
        if a < len(kpts) and b < len(kpts):
            ca = 1.0 if kpt_scores is None else float(kpt_scores[a])
            cb = 1.0 if kpt_scores is None else float(kpt_scores[b])
            if ca > 0.2 and cb > 0.2:
                xa, ya = int(kpts[a][0]), int(kpts[a][1])
                xb, yb = int(kpts[b][0]), int(kpts[b][1])
                ec = EDGE_COLORS[idx] if idx < len(EDGE_COLORS) else (255, 180, 0)
                cv2.line(frame, (xa, ya), (xb, yb), ec, 2, cv2.LINE_AA)

    for i, kp in enumerate(kpts):
        conf = 1.0 if kpt_scores is None else float(kpt_scores[i])
        if conf > 0.2:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 255, 255), -1)
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)


def draw_info_panel(frame, info: dict, pos=(10, 10)):
    """Vẽ bảng thông tin góc trên trái với nền mờ.

    Args:
        frame:  BGR numpy, modify in-place.
        info:   dict {key: value} hiển thị theo thứ tự.
        pos:    (x, y) góc trên trái của panel.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_h = 22
    pad = 8

    lines = [f"{k}: {v}" for k, v in info.items()]
    if not lines:
        return

    max_w = max(cv2.getTextSize(l, font, scale, thickness)[0][0] for l in lines)
    panel_h = line_h * len(lines) + pad * 2
    panel_w = max_w + pad * 2

    x0, y0 = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        ty = y0 + pad + line_h * i + line_h - 4
        cv2.putText(frame, line, (x0 + pad, ty), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)


def draw_reid_panel(frame, query_crop, matches: list, panel_w=220):
    """Vẽ panel ReID bên phải frame.

    Args:
        frame:       BGR frame.
        query_crop:  BGR crop của query person.
        matches:     list of (person_id: str, score: float, crop: np.ndarray | None)
        panel_w:     chiều rộng panel.
    Returns:
        frame mới có panel bên phải.
    """
    h, w = frame.shape[:2]
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_cursor = 10

    # Header
    cv2.putText(panel, "ReID Panel", (8, y_cursor + 14), font, 0.5, (200, 200, 200), 1)
    y_cursor += 30

    # Query
    cv2.putText(panel, "Query:", (8, y_cursor + 12), font, 0.45, (100, 220, 100), 1)
    y_cursor += 18
    if query_crop is not None and query_crop.size > 0:
        thumb_h = 80
        thumb_w = int(query_crop.shape[1] * thumb_h / max(query_crop.shape[0], 1))
        thumb_w = min(thumb_w, panel_w - 16)
        thumb = cv2.resize(query_crop, (thumb_w, thumb_h))
        if y_cursor + thumb_h < h:
            panel[y_cursor:y_cursor + thumb_h, 8:8 + thumb_w] = thumb
        y_cursor += thumb_h + 6

    # Matches
    cv2.putText(panel, f"Top {len(matches)} matches:", (8, y_cursor + 12), font, 0.42, (180, 180, 180), 1)
    y_cursor += 20

    for rank, (person_id, score, crop) in enumerate(matches):
        bar_color = (0, int(255 * score), int(255 * (1 - score)))
        # Score bar
        bar_len = int((panel_w - 16) * max(0.0, score))
        cv2.rectangle(panel, (8, y_cursor), (8 + bar_len, y_cursor + 14), bar_color, -1)
        label = f"#{rank+1} {person_id} {score:.2f}"
        cv2.putText(panel, label, (10, y_cursor + 11), font, 0.4, (0, 0, 0), 1)
        y_cursor += 18

        if crop is not None and crop.size > 0 and y_cursor + 70 < h:
            th = 70
            tw = min(int(crop.shape[1] * th / max(crop.shape[0], 1)), panel_w - 16)
            thumb = cv2.resize(crop, (tw, th))
            panel[y_cursor:y_cursor + th, 8:8 + tw] = thumb
            y_cursor += th + 4

        y_cursor += 4
        if y_cursor > h - 30:
            break

    return np.concatenate([frame, panel], axis=1)


class FPSCounter:
    """Tính FPS theo exponential moving average."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._t = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self._t
        self._t = now
        instant = 1.0 / max(dt, 1e-6)
        self.fps = self.alpha * instant + (1 - self.alpha) * self.fps
        return self.fps