from dataclasses import dataclass
from typing import List, Tuple, Protocol

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2)
    score: float
    label: str

class BaseDetector(Protocol):
    """Protocol for all detectors (Person, Object, Face)."""
    def detect(self, image) -> List[Detection]:
        ...
