from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class DetectionFilterStats:
    filtered_low_conf: int = 0
    filtered_small_box: int = 0
    filtered_bad_pose: int = 0
    filtered_non_person: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "filtered_low_conf": self.filtered_low_conf,
            "filtered_small_box": self.filtered_small_box,
            "filtered_bad_pose": self.filtered_bad_pose,
            "filtered_non_person": self.filtered_non_person,
        }


def bbox_area(bbox: Any) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def keypoint_quality(keypoint_scores: Any, min_keypoint_score: float) -> tuple[int, float]:
    if keypoint_scores is None:
        return 0, 0.0
    scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return 0, 0.0
    visible = int(np.count_nonzero(scores >= float(min_keypoint_score)))
    return visible, float(scores.mean())


def is_valid_person_detection(
    det: Mapping[str, Any],
    *,
    person_conf: float = 0.60,
    min_box_area: float = 2500,
    min_keypoints: int = 5,
    min_keypoint_score: float = 0.35,
) -> tuple[bool, str | None]:
    if int(det.get("class_id", -1)) != 0:
        return False, "filtered_non_person"

    if float(det.get("score", 0.0)) < float(person_conf):
        return False, "filtered_low_conf"

    if bbox_area(det.get("bbox", [0, 0, 0, 0])) < float(min_box_area):
        return False, "filtered_small_box"

    keypoints = det.get("keypoints")
    scores = det.get("keypoint_scores")
    if keypoints is not None and scores is not None:
        visible, avg_score = keypoint_quality(scores, min_keypoint_score)
        if visible < int(min_keypoints) or avg_score < float(min_keypoint_score):
            return False, "filtered_bad_pose"

    return True, None


def filter_person_detections(
    detections: list[dict[str, Any]],
    tracking_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], DetectionFilterStats]:
    stats = DetectionFilterStats()
    valid: list[dict[str, Any]] = []
    for det in detections:
        ok, reason = is_valid_person_detection(
            det,
            person_conf=float(tracking_cfg.get("person_conf", 0.60)),
            min_box_area=float(tracking_cfg.get("min_box_area", 2500)),
            min_keypoints=int(tracking_cfg.get("min_keypoints", 5)),
            min_keypoint_score=float(tracking_cfg.get("min_keypoint_score", 0.35)),
        )
        if ok:
            valid.append(det)
        elif reason and hasattr(stats, reason):
            setattr(stats, reason, getattr(stats, reason) + 1)
            if reason == "filtered_low_conf":
                stats.warnings.append(f"Warning: filtered false person conf={float(det.get('score', 0.0)):.2f}")
    return valid, stats
