from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    label: str = ""
    metadata: Dict[str, Any] = None

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Input: BGR Frame (numpy array)
        Output: List of Detection objects
        """
        pass
