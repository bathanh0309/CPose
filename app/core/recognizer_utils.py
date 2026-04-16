from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: np.ndarray
    last_seen_frame: int
    missed_frames: int = 0

@dataclass(slots=True)
class TrackedDetection:
    track_id: int
    bbox: np.ndarray
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    detection_conf: float

class SequentialTracker:
    """Tiny IoU-based tracker for stable local IDs within one clip."""
    def __init__(self, iou_threshold: float = 0.25, max_missed_frames: int = 15, center_distance_ratio: float = 0.18):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.center_distance_ratio = center_distance_ratio
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1

    def update(self, detections: List[Dict[str, Any]], frame_shape: tuple, frame_id: int) -> tuple[List[TrackedDetection], List[int]]:
        if not detections:
            expired = self._age_tracks(set())
            return [], expired

        frame_diag = math.hypot(float(frame_shape[1]), float(frame_shape[0]))
        matches = []
        for track_id, track in self._tracks.items():
            for det_idx, det in enumerate(detections):
                score = self._match_score(track.bbox, det["bbox"], frame_diag)
                if score is not None:
                    matches.append((score, track_id, det_idx))

        matches.sort(key=lambda x: x[0], reverse=True)
        assigned_tracks = set()
        assigned_dets = set()
        det_to_track = {}

        for score, track_id, det_idx in matches:
            if track_id in assigned_tracks or det_idx in assigned_dets:
                continue
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox = np.asarray(det["bbox"], dtype=float)
            track.last_seen_frame = frame_id
            track.missed_frames = 0
            assigned_tracks.add(track_id)
            assigned_dets.add(det_idx)
            det_to_track[det_idx] = track_id

        expired = self._age_tracks(assigned_tracks)

        for det_idx, det in enumerate(detections):
            if det_idx not in assigned_dets:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = TrackState(tid, np.asarray(det["bbox"], dtype=float), frame_id)
                det_to_track[det_idx] = tid

        tracked = []
        for det_idx, det in enumerate(detections):
            tracked.append(TrackedDetection(
                det_to_track[det_idx],
                np.asarray(det["bbox"], dtype=float),
                np.asarray(det["keypoints_xy"], dtype=float),
                np.asarray(det["keypoints_conf"], dtype=float),
                float(det.get("detection_conf", 0.0))
            ))
        return tracked, expired

    def _age_tracks(self, active_track_ids: set) -> List[int]:
        expired = []
        for tid in list(self._tracks.keys()):
            if tid not in active_track_ids:
                track = self._tracks[tid]
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    expired.append(tid)
                    del self._tracks[tid]
        return expired

    def _match_score(self, t_bbox, d_bbox, diag) -> Optional[float]:
        iou = self._bbox_iou(t_bbox, d_bbox)
        if iou >= self.iou_threshold: return 1.0 + iou
        dist = self._center_dist(t_bbox, d_bbox) / diag
        if dist <= self.center_distance_ratio: return 0.75 - dist
        return None

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1: return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_dist(box_a, box_b):
        acx, acy = (box_a[0]+box_a[2])/2, (box_a[1]+box_a[3])/2
        bcx, bcy = (box_b[0]+box_b[2])/2, (box_b[1]+box_b[3])/2
        return math.hypot(acx - bcx, acy - bcy)

class PoseTemporalSmoothing:
    def __init__(self, pose_ttl: int = 5):
        self.pose_ttl = pose_ttl
        self.track_memory: Dict[int, Dict] = {}

    def merge_pose(self, track_id: int, current_kps: np.ndarray) -> np.ndarray:
        memory = self.track_memory.get(track_id)
        if memory is None:
            self.track_memory[track_id] = {"kps": np.array(current_kps, copy=True), "ttl": self.pose_ttl}
            return current_kps

        prev_kps = memory["kps"]
        merged = []
        for cur, prev in zip(current_kps, prev_kps):
            cx, cy, cc = cur
            px, py, pc = prev
            if cc >= 0.35:
                mx, my, mc = 0.7*cx + 0.3*px, 0.7*cy + 0.3*py, cc
            elif memory["ttl"] > 0 and pc >= 0.35:
                mx, my, mc = px, py, pc * 0.9
            else:
                mx, my, mc = cx, cy, cc
            merged.append([mx, my, mc])

        merged_arr = np.array(merged, dtype=np.float32)
        memory["kps"] = merged_arr.copy()
        memory["ttl"] = self.pose_ttl if any(k[2] >= 0.35 for k in current_kps) else max(memory["ttl"] - 1, 0)
        return merged_arr

    def expire_track(self, tid: int):
        self.track_memory.pop(tid, None)

class ADLTemporalSmoothing:
    def __init__(self, hold_frames: int = 8, switch_margin: float = 0.08):
        self.hold_frames = hold_frames
        self.switch_margin = switch_margin
        self.track_state: Dict[int, Dict] = {}

    def smooth_adl(self, tid: int, raw_label: str, raw_conf: float) -> tuple[str, float]:
        state = self.track_state.get(tid)
        if state is None:
            self.track_state[tid] = {"label": raw_label, "conf": raw_conf, "hold": self.hold_frames}
            return raw_label, raw_conf
        if raw_label == state["label"]:
            state["conf"] = max(state["conf"], raw_conf)
            state["hold"] = self.hold_frames
            return state["label"], state["conf"]
        if raw_conf >= state["conf"] + self.switch_margin or state["hold"] <= 0:
            state["label"], state["conf"], state["hold"] = raw_label, raw_conf, self.hold_frames
        else:
            state["hold"] -= 1
        return state["label"], state["conf"]

    def expire_track(self, tid: int):
        self.track_state.pop(tid, None)
