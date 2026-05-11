"""Tracking implementations for CPose Module 2."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.human_detect.detector import resolve_detection_model


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}


def _get_yolo_model(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install requirements.txt") from exc
    if model_path not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if model_path not in _MODEL_CACHE:
                _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


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


def resolve_tracking_model(model: str | Path | None = None) -> str:
    return resolve_detection_model(model)


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
    return {
        **track,
        "age": int(age),
        "hits": int(hits),
        "misses": int(misses),
        "is_confirmed": bool(is_confirmed),
        "fragment_count": int(fragment_count),
        "quality_score": float(max(0.0, min(1.0, quality_score))),
        "failure_reason": "OK" if is_confirmed else "UNCONFIRMED_TRACK",
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
                if "prev_bbox" in track:
                    dx = float(track["bbox"][0]) - float(track["prev_bbox"][0])
                    dy = float(track["bbox"][1]) - float(track["prev_bbox"][1])
                    predicted_bbox = [
                        float(track["bbox"][0]) + dx,
                        float(track["bbox"][1]) + dy,
                        float(track["bbox"][2]) + dx,
                        float(track["bbox"][3]) + dy,
                    ]
                    score = max(score, iou(detection["bbox"], predicted_bbox))
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
            state = {"bbox": detection["bbox"], "missing": misses, "age": age, "hits": hits, "conf_sum": conf_sum}
            if previous.get("bbox") is not None:
                state["prev_bbox"] = previous["bbox"]
            self.tracks[best_id] = state
            outputs.append(enrich_track_metadata({"track_id": best_id, "bbox": detection["bbox"], "confidence": detection.get("confidence", 0.0), "class_name": "person"}, age, hits, misses, is_confirmed, quality_score=quality))
        for track_id in list(self.tracks):
            if track_id not in assigned_tracks:
                self.tracks[track_id]["missing"] += 1
                if self.tracks[track_id]["missing"] > self.max_missing:
                    del self.tracks[track_id]
        return outputs


class YoloByteTracker:
    def __init__(self, model_path: str | Path, conf: float = 0.5, tracker: str = "bytetrack.yaml", min_hits: int = 3, max_age: int = 30, window_size: int = 30) -> None:
        self.model = _get_yolo_model(str(model_path))
        self.conf = conf
        self.tracker = tracker
        self.min_hits = min_hits
        self.max_age = max_age
        self.window_size = window_size
        self.track_states: dict[int, dict] = {}
        self.frame_index = 0

    def reset(self) -> None:
        """Reset local and Ultralytics tracker state before a new video."""
        self.track_states.clear()
        self.frame_index = 0
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

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
            self.track_states[track_id] = {"age": age, "hits": hits, "misses": misses, "last_seen_frame": self.frame_index, "fragment_count": fragment_count, "last_bbox": bbox, "conf_sum": conf_sum}
            tracks.append(enrich_track_metadata({"track_id": track_id, "bbox": bbox, "confidence": confidence, "class_name": "person"}, age=age, hits=hits, misses=misses, is_confirmed=is_confirmed, fragment_count=fragment_count, quality_score=quality))
        self._age_missing(observed_ids)
        return tracks

    def _age_missing(self, observed_ids: set[int]) -> None:
        for track_id in list(self.track_states):
            if track_id in observed_ids:
                continue
            self.track_states[track_id]["misses"] = int(self.track_states[track_id].get("misses", 0)) + 1
            if self.track_states[track_id]["misses"] > self.max_age:
                del self.track_states[track_id]


__all__ = ["SimpleIoUTracker", "TrackMetadata", "YoloByteTracker", "bbox_iou", "enrich_track_metadata", "iou", "resolve_tracking_model"]
