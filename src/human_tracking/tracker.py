from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathlib import Path

from src.common.geometry import bbox_iou
from src.common.model_provider import get_yolo_model


@dataclass
class TrackMetadata:
    track_id: int
    bbox: list[float]
    confidence: float
    age: int
    hits: int
    misses: int
    is_confirmed: bool
    fragment_count: int
    quality_score: float
    failure_reason: str


def iou(box_a: list[float], box_b: list[float]) -> float:
    return bbox_iou(box_a, box_b)


def enrich_track_metadata(
    track: dict,
    age: int = 1,
    hits: int = 1,
    misses: int = 0,
    is_confirmed: bool = False,
    fragment_count: int = 0,
    quality_score: float | None = None,
) -> dict:
    if quality_score is None:
        quality_score = min(1.0, (hits / max(age, 1)) * (1 - min(misses, 5) / 5.0))
    failure_reason = "OK" if is_confirmed else "UNCONFIRMED_TRACK"
    return {
        **track,
        "age": int(age),
        "hits": int(hits),
        "misses": int(misses),
        "is_confirmed": bool(is_confirmed),
        "fragment_count": int(fragment_count),
        "quality_score": float(quality_score),
        "failure_reason": failure_reason,
    }


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 30, min_hits: int = 3, window_size: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.window_size = window_size
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        assigned_tracks: set[int] = set()
        outputs: list[dict] = []
        for detection in detections:
            best_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                score = iou(detection["bbox"], track["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1
            assigned_tracks.add(best_id)
            previous = self.tracks.get(best_id, {})
            age = int(previous.get("age", 0)) + 1
            hits = int(previous.get("hits", 0)) + 1
            misses = 0
            is_confirmed = hits >= self.min_hits and age >= self.min_hits
            conf_sum = float(previous.get("conf_sum", 0.0)) + float(detection.get("confidence", 0.0))
            avg_conf = conf_sum / max(hits, 1)
            quality = 0.5 * avg_conf + 0.3 * min(age / max(self.window_size, 1), 1.0)
            self.tracks[best_id] = {"bbox": detection["bbox"], "missing": misses, "age": age, "hits": hits, "conf_sum": conf_sum}
            outputs.append(enrich_track_metadata({
                "track_id": best_id,
                "bbox": detection["bbox"],
                "confidence": detection.get("confidence", 0.0),
                "class_name": "person",
            }, age, hits, misses, is_confirmed, quality_score=quality))
        for track_id in list(self.tracks):
            if track_id not in assigned_tracks:
                self.tracks[track_id]["missing"] += 1
                if self.tracks[track_id]["missing"] > self.max_missing:
                    del self.tracks[track_id]
        return outputs


class YoloByteTracker:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.5,
        tracker: str = "bytetrack.yaml",
        min_hits: int = 3,
        max_age: int = 30,
        window_size: int = 30,
    ) -> None:
        self.model = get_yolo_model(str(model_path))
        self.conf = conf
        self.tracker = tracker
        self.min_hits = min_hits
        self.max_age = max_age
        self.window_size = window_size
        self.track_states: dict[int, dict] = {}
        self.frame_index = 0

    def track(self, frame: Any) -> list[dict]:
        self.frame_index += 1
        results = self.model.track(frame, conf=self.conf, classes=[0], tracker=self.tracker, persist=True, verbose=False)
        tracks: list[dict] = []
        observed_ids: set[int] = set()
        if not results or results[0].boxes is None:
            self._age_missing(set())
            return tracks
        for box in results[0].boxes:
            if box.id is None:
                continue
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            if class_id != 0:
                continue
            track_id = int(box.id[0].detach().cpu().item())
            observed_ids.add(track_id)
            bbox = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            state = self.track_states.get(track_id, {})
            previous_misses = int(state.get("misses", 0))
            fragment_count = int(state.get("fragment_count", 0)) + (1 if previous_misses > 0 else 0)
            age = int(state.get("age", 0)) + 1
            hits = int(state.get("hits", 0)) + 1
            conf_sum = float(state.get("conf_sum", 0.0)) + confidence
            misses = 0
            avg_conf = conf_sum / max(hits, 1)
            quality = 0.5 * avg_conf + 0.3 * min(age / max(self.window_size, 1), 1.0) - 0.2 * min(misses / max(self.max_age, 1), 1.0)
            is_confirmed = hits >= self.min_hits and age >= self.min_hits
            self.track_states[track_id] = {
                "age": age,
                "hits": hits,
                "misses": misses,
                "last_seen_frame": self.frame_index,
                "fragment_count": fragment_count,
                "last_bbox": bbox,
                "conf_sum": conf_sum,
            }
            tracks.append(enrich_track_metadata({
                "track_id": track_id,
                "bbox": bbox,
                "confidence": confidence,
                "class_name": "person",
            }, age=age, hits=hits, misses=misses, is_confirmed=is_confirmed, fragment_count=fragment_count, quality_score=max(0.0, min(1.0, quality))))
        self._age_missing(observed_ids)
        return tracks

    def _age_missing(self, observed_ids: set[int]) -> None:
        for track_id in list(self.track_states):
            if track_id in observed_ids:
                continue
            self.track_states[track_id]["misses"] = int(self.track_states[track_id].get("misses", 0)) + 1
            if self.track_states[track_id]["misses"] > self.max_age:
                del self.track_states[track_id]
