"""
TrackCache — anti-flicker bbox cache for stable tracking display.

When the detector misses a person for a few frames, we keep drawing the last
known bbox until ``ttl_frames`` is exceeded. This prevents the common
"blinking bbox" problem in real-time tracking systems.
"""

from __future__ import annotations

from typing import Any


class TrackCache:
    """Cache recent track detections to prevent bbox flickering."""

    def __init__(self, ttl_frames: int = 25):
        self.ttl_frames = int(ttl_frames)
        self.tracks: dict[int, dict[str, Any]] = {}

    def update(self, detections: list[dict[str, Any]], frame_idx: int) -> None:
        """Update cache with fresh detections from the current frame."""
        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                # Generate a stable pseudo-ID from quantised bbox position
                x1, y1, x2, y2 = map(int, det["bbox"])
                tid = hash((x1 // 20, y1 // 20, x2 // 20, y2 // 20)) % 100000

            self.tracks[tid] = {
                "track_id": tid,
                "bbox": det["bbox"],
                "score": float(det.get("score", 0.0)),
                "last_seen": int(frame_idx),
                "det": det,
            }

    def active(self, frame_idx: int) -> list[dict[str, Any]]:
        """Return all tracks still within their TTL window."""
        alive: list[dict[str, Any]] = []
        expired: list[int] = []

        for tid, item in self.tracks.items():
            age = int(frame_idx) - int(item["last_seen"])
            if age <= self.ttl_frames:
                alive.append({
                    **item["det"],
                    "track_id": tid,
                    "bbox": item["bbox"],
                    "score": item["score"],
                    "cached": age > 0,
                    "age": age,
                })
            else:
                expired.append(tid)

        for tid in expired:
            self.tracks.pop(tid, None)

        return alive

    def clear(self) -> None:
        self.tracks.clear()
