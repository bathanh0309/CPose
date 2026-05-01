from __future__ import annotations

from src.common.geometry import bbox_iou


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    return bbox_iou(box1, box2)

