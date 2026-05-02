"""Timing helpers shared by CPose modules."""
from __future__ import annotations

import time


class Timer:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def fps(processed_frames: int, elapsed_sec: float) -> float:
    return processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0


def latency_ms(processed_frames: int, elapsed_sec: float) -> float:
    return (elapsed_sec / processed_frames * 1000.0) if processed_frames > 0 else 0.0


__all__ = ["Timer", "fps", "latency_ms"]
