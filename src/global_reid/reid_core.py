from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.common.topology import CameraTopology, infer_zone, topology_score
from src.global_reid.body_features import cosine_similarity, extract_body_hsv_feature, extract_height_ratio, histogram_similarity
from src.global_reid.fusion_score import DEFAULT_WEIGHTS, weighted_fusion
from src.global_reid.state_machine import ACTIVE, CLOTHING_CHANGE_SUSPECTED, DORMANT, next_missing_state


@dataclass
class GlobalPerson:
    gid: int
    state: str = ACTIVE
    last_camera_id: str = ""
    last_time: datetime | None = None
    last_bbox: list[float] = field(default_factory=list)
    last_exit_zone: str | None = None
    last_entry_zone: str | None = None
    last_adl: str | None = None
    last_body_feat: Any = None
    last_face_embedding: list[float] | None = None
    last_pose_signature: list[float] | None = None
    last_height_ratio: float | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def global_id(self) -> str:
        return f"GID-{self.gid:03d}"


def pose_signature(keypoints: list[dict] | None, bbox: list[float]) -> list[float] | None:
    if not keypoints or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    values: list[float] = []
    for keypoint in keypoints:
        conf = float(keypoint.get("confidence", 0.0))
        if conf < 0.3:
            values.extend([0.0, 0.0])
        else:
            values.extend([(float(keypoint.get("x", 0.0)) - x1) / width, (float(keypoint.get("y", 0.0)) - y1) / height])
    return values


class GlobalPersonTable:
    def __init__(self) -> None:
        self._table: dict[int, GlobalPerson] = {}
        self._next_gid = 1
        self.counts = {
            "new_id_count": 0,
            "strong_match_count": 0,
            "soft_match_count": 0,
            "ambiguous_match_count": 0,
            "topology_conflict_count": 0,
            "multi_candidate_conflict_count": 0,
        }

    def _new_person(self) -> GlobalPerson:
        person = GlobalPerson(gid=self._next_gid)
        self._table[person.gid] = person
        self._next_gid += 1
        self.counts["new_id_count"] += 1
        return person

    def _candidate_scores(
        self,
        person: GlobalPerson,
        bbox: list[float],
        frame: Any,
        camera_id: str,
        current_time: datetime | None,
        keypoints: list[dict] | None,
        face_event: dict | None,
        topology: CameraTopology,
        config: dict,
    ) -> dict[str, Any]:
        delta = (current_time - person.last_time).total_seconds() if current_time and person.last_time else None
        entry_zone = infer_zone(camera_id, bbox, "entry", topology)
        topo = topology_score(topology, person.last_camera_id, camera_id, delta if delta is not None else 0.0, person.last_exit_zone, entry_zone)
        body_feat = extract_body_hsv_feature(frame, bbox)
        height_ratio = extract_height_ratio(bbox, frame.shape[0] if frame is not None else None)
        pose_feat = pose_signature(keypoints, bbox)
        face_embedding = face_event.get("embedding") if face_event else None
        score_face = cosine_similarity(face_embedding, person.last_face_embedding)
        score_body = histogram_similarity(body_feat, person.last_body_feat)
        score_pose = cosine_similarity(pose_feat, person.last_pose_signature)
        score_height = None if height_ratio is None or person.last_height_ratio is None else max(0.0, 1.0 - abs(height_ratio - person.last_height_ratio) * 3.0)
        max_age = float(config.get("max_candidate_age_sec", 300))
        score_time = None if delta is None else max(0.0, 1.0 - min(delta, max_age) / max(max_age, 1.0))
        score_topology = float(topo["score"]) if topo.get("allowed") or topo.get("failure_reason") != "TOPOLOGY_CONFLICT" else 0.0
        mode = "normal" if score_face is not None else "no_face"
        weights = (config.get("weights") or DEFAULT_WEIGHTS).get(mode, DEFAULT_WEIGHTS[mode])
        total = weighted_fusion({
            "face": score_face,
            "body": score_body,
            "pose": score_pose,
            "height": score_height,
            "time": score_time,
            "topology": score_topology,
        }, weights)
        return {
            "person": person,
            "score_total": total,
            "score_face": score_face,
            "score_body": score_body,
            "score_pose": score_pose,
            "score_height": score_height,
            "score_time": score_time,
            "score_topology": score_topology,
            "topology_allowed": bool(topo.get("allowed")),
            "delta_time_sec": delta,
            "entry_zone": entry_zone,
            "exit_zone": person.last_exit_zone,
            "failure_reason": topo.get("failure_reason", "OK"),
            "body_feat": body_feat,
            "height_ratio": height_ratio,
            "pose_signature": pose_feat,
            "face_embedding": face_embedding,
        }

    def match_or_create(
        self,
        bbox: list[float],
        frame: Any,
        camera_id: str,
        current_time: datetime | None,
        track_id: int,
        adl_label: str | None,
        keypoints: list[dict] | None,
        face_event: dict | None,
        topology: CameraTopology,
        config: dict,
    ) -> tuple[GlobalPerson, dict]:
        candidates = [p for p in self._table.values() if p.state != "CLOSED"]
        scored = [self._candidate_scores(p, bbox, frame, camera_id, current_time, keypoints, face_event, topology, config) for p in candidates]
        viable = [row for row in scored if row["score_total"] is not None and row["topology_allowed"]]
        viable.sort(key=lambda row: float(row["score_total"]), reverse=True)
        strong = float(config.get("strong_threshold", 0.65))
        weak = float(config.get("weak_threshold", 0.45))
        margin = float(config.get("ambiguous_margin", 0.05))
        if not viable:
            person = self._new_person()
            match_status = "new_id" if not scored else "no_candidate"
            best = self._fresh_scores(bbox, frame, keypoints, face_event, topology, camera_id)
            if any(row.get("failure_reason") == "TOPOLOGY_CONFLICT" for row in scored):
                self.counts["topology_conflict_count"] += 1
                best["failure_reason"] = "TOPOLOGY_CONFLICT"
        else:
            best = viable[0]
            second = viable[1] if len(viable) > 1 else None
            if second and abs(float(best["score_total"]) - float(second["score_total"])) < margin:
                person = best["person"]
                match_status = "ambiguous"
                best["failure_reason"] = "MULTI_CANDIDATE_CONFLICT"
                self.counts["ambiguous_match_count"] += 1
                self.counts["multi_candidate_conflict_count"] += 1
            elif float(best["score_total"]) >= strong:
                person = best["person"]
                match_status = "strong_match"
                self.counts["strong_match_count"] += 1
            elif float(best["score_total"]) >= weak:
                person = best["person"]
                match_status = "soft_match"
                self.counts["soft_match_count"] += 1
            else:
                person = self._new_person()
                match_status = "new_id"
                best = self._fresh_scores(bbox, frame, keypoints, face_event, topology, camera_id)

        self._update_person(person, bbox, camera_id, current_time, adl_label, best, topology)
        match_info = {k: best.get(k) for k in (
            "score_total", "score_face", "score_body", "score_pose", "score_height", "score_time", "score_topology",
            "topology_allowed", "delta_time_sec", "entry_zone", "exit_zone", "failure_reason"
        )}
        match_info.update({"match_status": match_status, "state": person.state})
        person.history.append({"camera_id": camera_id, "time": current_time.isoformat() if current_time else None, "track_id": track_id, "adl": adl_label, **match_info})
        return person, match_info

    def _fresh_scores(self, bbox: list[float], frame: Any, keypoints: list[dict] | None, face_event: dict | None, topology: CameraTopology, camera_id: str) -> dict[str, Any]:
        return {
            "score_total": None,
            "score_face": None,
            "score_body": None,
            "score_pose": None,
            "score_height": None,
            "score_time": None,
            "score_topology": None,
            "topology_allowed": None,
            "delta_time_sec": None,
            "entry_zone": infer_zone(camera_id, bbox, "entry", topology),
            "exit_zone": None,
            "failure_reason": "OK",
            "body_feat": extract_body_hsv_feature(frame, bbox),
            "height_ratio": extract_height_ratio(bbox, frame.shape[0] if frame is not None else None),
            "pose_signature": pose_signature(keypoints, bbox),
            "face_embedding": face_event.get("embedding") if face_event else None,
        }

    def _update_person(self, person: GlobalPerson, bbox: list[float], camera_id: str, current_time: datetime | None, adl_label: str | None, scores: dict, topology: CameraTopology) -> None:
        person.state = CLOTHING_CHANGE_SUSPECTED if scores.get("score_body") is not None and scores.get("score_body", 1.0) < 0.25 and scores.get("score_height", 0.0) > 0.7 else ACTIVE
        person.last_camera_id = camera_id
        person.last_time = current_time
        person.last_bbox = bbox
        person.last_entry_zone = scores.get("entry_zone")
        person.last_exit_zone = infer_zone(camera_id, bbox, "exit", topology)
        person.last_adl = adl_label
        person.last_body_feat = scores.get("body_feat")
        person.last_height_ratio = scores.get("height_ratio")
        person.last_pose_signature = scores.get("pose_signature")
        person.last_face_embedding = scores.get("face_embedding")

    def mark_dormant_missing(self, active_gids: set[int], current_time: datetime | None, max_candidate_age_sec: float = 300.0) -> None:
        if current_time is None:
            return
        for person in self._table.values():
            if person.gid in active_gids or person.last_time is None:
                continue
            missing = (current_time - person.last_time).total_seconds()
            person.state = next_missing_state(person.state, missing, max_candidate_age_sec)

    def to_dict(self) -> dict[str, Any]:
        return {
            person.global_id: {
                "state": person.state if person.state != DORMANT else DORMANT,
                "last_camera_id": person.last_camera_id,
                "last_time": person.last_time.isoformat() if person.last_time else None,
                "last_adl": person.last_adl,
                "history": person.history,
            }
            for person in self._table.values()
        }
