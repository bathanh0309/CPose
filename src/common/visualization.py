from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4),
]


def draw_bbox(frame: np.ndarray, bbox: Iterable[float], label: str, color: tuple[int, int, int] = (0, 220, 0)) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_track(frame: np.ndarray, bbox: Iterable[float], track_id: int, confidence: float) -> None:
    """Local track (module 2). Shows local track ID (NOT global cross-camera ID)."""
    draw_bbox(frame, bbox, f"T{track_id} {confidence:.2f}", (255, 180, 0))


def draw_skeleton(frame: np.ndarray, keypoints: list[dict], min_conf: float = 0.25) -> None:
    points: dict[int, tuple[int, int]] = {}
    for keypoint in keypoints:
        if float(keypoint.get("confidence", 0.0)) >= min_conf:
            points[int(keypoint["id"])] = (int(keypoint["x"]), int(keypoint["y"]))
    for start, end in COCO_EDGES:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (0, 180, 255), 2)
    for point in points.values():
        cv2.circle(frame, point, 3, (0, 255, 255), -1)


def draw_adl_label(frame: np.ndarray, bbox: Iterable[float], track_id: int, label: str, confidence: float) -> None:
    """ADL label (module 4). Shows ADL activity only - NO Global ID (that is ReID's job)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw_bbox(frame, (x1, y1, x2, y2), "", (80, 180, 255))
    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 180, 255),
        2,
    )


def draw_global_id(
    frame: np.ndarray,
    bbox: Iterable[float],
    global_id: int,
    adl_label: str = "",
    status: str = "ACTIVE",
) -> None:
    """Cross-camera Global ID visualization — MODULE 5 (ReID) ONLY.

    This is the ONLY function that renders a permanent, cross-camera GID-XXX label
    on the video overlay.  Modules 1–4 must NOT render GID labels.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color_map = {
        "ACTIVE": (0, 255, 80),
        "PENDING_TRANSFER": (255, 200, 0),
        "IN_ROOM": (0, 180, 255),
        "DORMANT": (160, 160, 160),
        "SOFT_MATCH": (200, 100, 255),
        "CLOSED": (100, 100, 100),
    }
    color = color_map.get(status, (0, 255, 80))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"GID-{global_id:03d}"
    if adl_label:
        label += f" | {adl_label}"
    cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # status badge (bottom-left of bbox)
    cv2.putText(frame, status, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
