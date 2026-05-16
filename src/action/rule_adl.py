import numpy as np


LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


def _center(points, scores, joints, min_score=0.2):
    valid = [j for j in joints if j < len(points) and scores[j] >= min_score]
    if not valid:
        return None
    return points[valid].mean(axis=0)


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cos_val = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _median_leg_angle(points, scores):
    angles = []
    for hip, knee, ankle in ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)):
        if min(scores[hip], scores[knee], scores[ankle]) < 0.2:
            continue
        angle = _angle(points[hip], points[knee], points[ankle])
        if angle is not None:
            angles.append(angle)
    if not angles:
        return None
    return float(np.median(angles))


def classify_rule_adl(window):
    """Classify coarse ADL from COCO-17 keypoint geometry.

    This is a fallback for live demos when the PoseC3D stack or pkl inference is
    unavailable. It intentionally returns only the three coarse labels used by
    the app: standing, sitting, walking.
    """
    if not window:
        return {"status": "failed", "label": "unknown", "score": 0.0, "method": "rule"}

    keypoints = np.asarray(window.get("keypoints"), dtype=np.float32)
    scores = np.asarray(window.get("scores"), dtype=np.float32)
    if keypoints.ndim != 3 or scores.ndim != 2 or keypoints.shape[:2] != scores.shape:
        return {"status": "failed", "label": "unknown", "score": 0.0, "method": "rule"}

    visible = scores[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]].mean()
    if visible < 0.2:
        return {"status": "failed", "label": "unknown", "score": 0.0, "method": "rule"}

    centers = []
    ankle_gaps = []
    body_heights = []
    for pts, conf in zip(keypoints, scores):
        hip_center = _center(pts, conf, (LEFT_HIP, RIGHT_HIP))
        shoulder_center = _center(pts, conf, (LEFT_SHOULDER, RIGHT_SHOULDER))
        left_ankle = pts[LEFT_ANKLE] if conf[LEFT_ANKLE] >= 0.2 else None
        right_ankle = pts[RIGHT_ANKLE] if conf[RIGHT_ANKLE] >= 0.2 else None
        if hip_center is not None:
            centers.append(hip_center)
        if left_ankle is not None and right_ankle is not None:
            ankle_gaps.append(abs(float(left_ankle[0] - right_ankle[0])))
        valid_pts = pts[conf >= 0.2]
        if len(valid_pts) >= 4:
            body_heights.append(float(valid_pts[:, 1].max() - valid_pts[:, 1].min()))

    if not body_heights:
        return {"status": "failed", "label": "unknown", "score": 0.0, "method": "rule"}

    body_h = max(float(np.median(body_heights)), 1.0)
    movement = 0.0
    if len(centers) >= 2:
        centers = np.asarray(centers)
        displacement = np.linalg.norm(centers[-1] - centers[0])
        step_delta = np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()
        movement = max(displacement, step_delta * 0.35) / body_h

    last_pts = keypoints[-1]
    last_scores = scores[-1]
    shoulder = _center(last_pts, last_scores, (LEFT_SHOULDER, RIGHT_SHOULDER))
    hip = _center(last_pts, last_scores, (LEFT_HIP, RIGHT_HIP))
    knee = _center(last_pts, last_scores, (LEFT_KNEE, RIGHT_KNEE))
    leg_angle = _median_leg_angle(last_pts, last_scores)

    torso_ratio = None
    knee_drop = None
    if shoulder is not None and hip is not None:
        torso_ratio = abs(float(hip[1] - shoulder[1])) / body_h
    if hip is not None and knee is not None:
        knee_drop = abs(float(knee[1] - hip[1])) / body_h

    ankle_gap = float(np.median(ankle_gaps)) / body_h if ankle_gaps else 0.0

    if movement > 0.18 or (movement > 0.10 and ankle_gap > 0.18):
        score = min(0.95, 0.55 + movement)
        return {"status": "inferred", "label": "walking", "score": score, "method": "rule"}

    sitting_votes = 0
    if leg_angle is not None and leg_angle < 145:
        sitting_votes += 1
    if torso_ratio is not None and torso_ratio < 0.42:
        sitting_votes += 1
    if knee_drop is not None and knee_drop < 0.32:
        sitting_votes += 1

    if sitting_votes >= 2:
        return {"status": "inferred", "label": "sitting", "score": 0.70, "method": "rule"}

    return {"status": "inferred", "label": "standing", "score": 0.65, "method": "rule"}
