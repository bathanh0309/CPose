"""Multi-frame fall detection helper."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class FallDetector:
    min_falling_frames: int = 2
    window_size: int = 5
    history: deque[str] = field(default_factory=deque)

    def update(self, adl_label: str) -> bool:
        if self.history.maxlen != self.window_size:
            self.history = deque(self.history, maxlen=self.window_size)
        self.history.append(adl_label)
        return sum(1 for label in self.history if label == "falling") >= self.min_falling_frames

    def reset(self) -> None:
        self.history.clear()


__all__ = ["FallDetector"]
