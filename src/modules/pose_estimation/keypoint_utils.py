"""Pose keypoint helpers."""
from __future__ import annotations

import numpy as np


def normalize_keypoints(keypoints, center_joint_idx: int = 11):
    center = keypoints[:, center_joint_idx:center_joint_idx + 1, :]
    keypoints = keypoints - center
    scale = np.max(np.linalg.norm(keypoints, axis=-1))
    return keypoints / scale if scale > 0 else keypoints


def get_bone_features(keypoints, skeleton_type: str = "coco"):
    return None


def resample_sequence(keypoints, target_len: int = 30):
    if keypoints.shape[0] == target_len:
        return keypoints
    indices = np.linspace(0, keypoints.shape[0] - 1, target_len).astype(int)
    return keypoints[indices]


__all__ = ["get_bone_features", "normalize_keypoints", "resample_sequence"]
