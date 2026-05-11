"""Pose estimator facade."""
from __future__ import annotations

from models.pose_estimation.pose_model import COCO_KEYPOINT_NAMES, PoseModel as PoseEstimator, resolve_pose_model

__all__ = ["COCO_KEYPOINT_NAMES", "PoseEstimator", "resolve_pose_model"]
