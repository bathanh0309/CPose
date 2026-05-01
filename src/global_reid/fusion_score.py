from __future__ import annotations

from dataclasses import dataclass


DEFAULT_WEIGHTS = {
    "normal": {"face": 0.30, "body": 0.20, "pose": 0.15, "height": 0.10, "time": 0.15, "topology": 0.10},
    "no_face": {"body": 0.30, "pose": 0.20, "height": 0.15, "time": 0.20, "topology": 0.15},
    "clothing_change_suspected": {"face": 0.35, "body": 0.05, "pose": 0.20, "height": 0.15, "time": 0.15, "topology": 0.10},
}


@dataclass(slots=True)
class CandidateScores:
    score_total: float | None
    score_face: float | None
    score_body: float | None
    score_pose: float | None
    score_height: float | None
    score_time: float | None
    score_topology: float | None
    topology_allowed: bool | None
    delta_time_sec: float | None
    entry_zone: str | None
    exit_zone: str | None
    failure_reason: str


def weighted_fusion(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    usable = {key: value for key, value in scores.items() if value is not None and key in weights}
    if not usable:
        return None
    weight_total = sum(float(weights[key]) for key in usable)
    if weight_total <= 0:
        return None
    return sum(float(value) * float(weights[key]) / weight_total for key, value in usable.items())
