from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class PoseResult:
    keypoints: np.ndarray  # Shape (17, 3) - [x, y, conf]
    bbox: List[float]      # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = None

class BasePoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Input: BGR Frame (numpy array)
        Output: List of PoseResult objects
        """
        pass
