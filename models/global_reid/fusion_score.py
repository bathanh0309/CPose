"""Compatibility exports for the score fusion helpers."""
from __future__ import annotations

from models.global_reid.matching import (
    CandidateScores,
    DEFAULT_WEIGHTS,
    cosine_similarity,
    extract_body_hsv_feature,
    extract_height_ratio,
    histogram_similarity,
    pose_signature,
    weighted_fusion,
)

__all__ = [
    "CandidateScores",
    "DEFAULT_WEIGHTS",
    "cosine_similarity",
    "extract_body_hsv_feature",
    "extract_height_ratio",
    "histogram_similarity",
    "pose_signature",
    "weighted_fusion",
]
