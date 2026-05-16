"""One Euro filters for smoothing realtime pose keypoints."""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """Smooth a scalar signal with an adaptive low-pass filter."""

    def __init__(
        self,
        freq: float = 25.0,
        mincutoff: float = 1.0,
        beta: float = 0.1,
        dcutoff: float = 1.0,
    ) -> None:
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x: float | None = None
        self._dx: float | None = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * float(cutoff))
        return 1.0 / (1.0 + tau / (1.0 / self.freq))

    def __call__(self, x: float) -> float:
        value = float(x)
        if self._x is None:
            self._x = value
            self._dx = 0.0
            return value
        dx = (value - self._x) * self.freq
        prev_dx = 0.0 if self._dx is None else self._dx
        edx = prev_dx + self._alpha(self.dcutoff) * (dx - prev_dx)
        cutoff = self.mincutoff + self.beta * abs(edx)
        self._x = self._x + self._alpha(cutoff) * (value - self._x)
        self._dx = edx
        return self._x


class KeypointSmoother:
    """Maintain per-track One Euro filters for COCO-17 keypoints."""

    def __init__(self, freq: float = 25.0) -> None:
        self.freq = float(freq)
        self.filters: dict[int, list[OneEuroFilter]] = {}

    def smooth(self, track_id: int, kps: np.ndarray) -> np.ndarray:
        """Smooth [17,3] keypoints, preserving confidence scores."""
        tid = int(track_id)
        arr = np.asarray(kps, dtype=np.float32)
        if arr.shape != (17, 3):
            return arr
        if tid not in self.filters:
            self.filters[tid] = [OneEuroFilter(freq=self.freq) for _ in range(34)]
        filters = self.filters[tid]
        out = arr.copy()
        for idx in range(17):
            out[idx, 0] = filters[idx * 2](float(arr[idx, 0]))
            out[idx, 1] = filters[idx * 2 + 1](float(arr[idx, 1]))
        return out

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Drop filters for tracks that are no longer active."""
        active = {int(tid) for tid in active_track_ids}
        for tid in list(self.filters):
            if tid not in active:
                self.filters.pop(tid, None)
