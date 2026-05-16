from __future__ import annotations

import numpy as np


class ByteTrackWrapper:
    """Thin wrapper for Ultralytics ByteTrack via YoloPoseTracker.

    The actual tracker state lives inside YOLO.track(..., persist=True). This
    class exists to keep the module boundary explicit without introducing an
    external ByteTrack checkpoint dependency.
    """

    def __init__(self, detector):
        if not hasattr(detector, "infer"):
            raise TypeError("ByteTrackWrapper expects a detector with infer(frame, persist=True)")
        self.detector = detector

    def update(self, frame):
        """Return (detections, raw_result)."""
        return self.detector.infer(frame, persist=True)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    xx1 = np.maximum(ax1[:, None], bx1[None])
    yy1 = np.maximum(ay1[:, None], by1[None])
    xx2 = np.minimum(ax2[:, None], bx2[None])
    yy2 = np.minimum(ay2[:, None], by2[None])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


class ByteTrackNumpy:
    """CPU-only IoU tracker with ByteTrack-style high/low score association."""

    def __init__(self, high_thresh: float = 0.4, low_thresh: float = 0.1, match_thresh: float = 0.3, max_age: int = 30):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.max_age = int(max_age)
        self.next_id = 1
        self.tracks: dict[int, dict] = {}
        self.frame_idx = 0

    def update(self, bboxes: list[list[float]], frame=None) -> list[dict]:
        self.frame_idx += 1
        dets = np.asarray(bboxes, dtype=np.float32).reshape(-1, 5) if bboxes else np.empty((0, 5), dtype=np.float32)
        outputs: list[dict] = []
        if len(dets) == 0:
            self._prune()
            return outputs

        track_ids = list(self.tracks.keys())
        track_boxes = np.asarray([self.tracks[tid]["bbox"] for tid in track_ids], dtype=np.float32).reshape(-1, 4)
        assigned_det: set[int] = set()
        assigned_track: set[int] = set()

        for det_indices in (np.where(dets[:, 4] >= self.high_thresh)[0], np.where((dets[:, 4] < self.high_thresh) & (dets[:, 4] >= self.low_thresh))[0]):
            if len(det_indices) == 0 or len(track_ids) == 0:
                continue
            ious = _iou_matrix(track_boxes, dets[det_indices, :4])
            while ious.size and float(np.max(ious)) >= self.match_thresh:
                tr_pos, det_pos = np.unravel_index(int(np.argmax(ious)), ious.shape)
                tid = track_ids[int(tr_pos)]
                di = int(det_indices[int(det_pos)])
                if tid in assigned_track or di in assigned_det:
                    ious[tr_pos, det_pos] = -1
                    continue
                self._update_track(tid, dets[di])
                outputs.append(self._as_detection(tid))
                assigned_track.add(tid)
                assigned_det.add(di)
                ious[tr_pos, :] = -1
                ious[:, det_pos] = -1

        for di, det in enumerate(dets):
            if di in assigned_det or float(det[4]) < self.high_thresh:
                continue
            tid = self.next_id
            self.next_id += 1
            self._update_track(tid, det)
            outputs.append(self._as_detection(tid))

        self._prune()
        return outputs

    def _update_track(self, track_id: int, det: np.ndarray) -> None:
        self.tracks[int(track_id)] = {
            "bbox": det[:4].astype(np.float32),
            "score": float(det[4]),
            "last_seen": self.frame_idx,
        }

    def _as_detection(self, track_id: int) -> dict:
        tr = self.tracks[int(track_id)]
        return {
            "track_id": int(track_id),
            "bbox": tr["bbox"].astype(float).tolist(),
            "score": float(tr["score"]),
            "class_id": 0,
        }

    def _prune(self) -> None:
        for tid, tr in list(self.tracks.items()):
            if self.frame_idx - int(tr["last_seen"]) > self.max_age:
                self.tracks.pop(tid, None)
