"""
TrackCache — anti-flicker bbox cache for stable tracking display.

FIXED:
- No longer creates pseudo-IDs from bbox hashes for track_id=-1 detections.
  Ghost tracks caused the "tracked=5" problem when only 1 person was present.
- Adds min_conf_for_cache: weak detections are not cached.
- Adds max_cached_tracks: hard cap so the cache cannot grow unboundedly.
- Only live (non-cached) detections should be fed to pose / ADL / ReID / gallery.
  The caller must separate `fresh_detections` (from tracker) from `display_dets`
  (from TrackCache.active()) and use fresh ones for AI inference.
"""

from __future__ import annotations

from typing import Any


class TrackCache:
    """Cache recent track detections to prevent bbox flickering.

    Parameters
    ----------
    ttl_frames : int
        How many frames a track is kept after it was last seen (default 12).
    min_conf_for_cache : float
        Only cache detections above this confidence (default 0.45).
    max_cached_tracks : int
        Hard cap on the number of simultaneously cached tracks (default 5).
        Oldest tracks are evicted first when the cap is exceeded.
    """

    def __init__(
        self,
        ttl_frames: int = 12,
        min_conf_for_cache: float = 0.45,
        max_cached_tracks: int = 5,
    ):
        self.ttl_frames = int(ttl_frames)
        self.min_conf_for_cache = float(min_conf_for_cache)
        self.max_cached_tracks = int(max_cached_tracks)
        self.tracks: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def update(self, detections: list[dict[str, Any]], frame_idx: int) -> None:
        """Update cache with *fresh* (non-cached) detections from the current frame.

        Rules:
        * track_id must be >= 0  (no hash-based pseudo IDs for -1 tracks)
        * detection confidence must be >= min_conf_for_cache
        * when the cache exceeds max_cached_tracks the oldest tracks are removed
        """
        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue  # skip untracked detections – no ghost IDs

            score = float(det.get("score", 0.0))
            if score < self.min_conf_for_cache:
                continue  # skip weak detections

            self.tracks[tid] = {
                "track_id": tid,
                "bbox": det["bbox"],
                "score": score,
                "last_seen": int(frame_idx),
                "det": det,
            }

        # Enforce max_cached_tracks – keep the most recently seen tracks
        if len(self.tracks) > self.max_cached_tracks:
            sorted_ids = sorted(
                self.tracks,
                key=lambda t: self.tracks[t]["last_seen"],
                reverse=True,
            )
            self.tracks = {tid: self.tracks[tid] for tid in sorted_ids[: self.max_cached_tracks]}

    # ------------------------------------------------------------------
    def active(self, frame_idx: int) -> list[dict[str, Any]]:
        """Return all tracks still within their TTL window.

        The returned dicts carry a ``cached`` flag (True when the track was
        not seen in the *current* frame).  Callers must NOT feed cached
        detections to pose estimation, ADL, ReID, or gallery events.
        """
        alive: list[dict[str, Any]] = []
        expired: list[int] = []

        for tid, item in self.tracks.items():
            age = int(frame_idx) - int(item["last_seen"])
            if age <= self.ttl_frames:
                alive.append(
                    {
                        **item["det"],
                        "track_id": tid,
                        "bbox": item["bbox"],
                        "score": item["score"],
                        "cached": age > 0,   # True → stale, don't feed to AI
                        "age": age,
                    }
                )
            else:
                expired.append(tid)

        for tid in expired:
            self.tracks.pop(tid, None)

        return alive

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.tracks.clear()
