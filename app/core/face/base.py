from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

class BaseFaceRecognizer(ABC):
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces and return list of cropped face images or bboxes"""
        pass

    @abstractmethod
    def encode(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding vector (e.g., 512-d) from face image"""
        pass

    @abstractmethod
    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compare two embeddings and return similarity score (0.0 to 1.0)"""
        pass

    @abstractmethod
    def register(self, name: str, embeddings: List[np.ndarray]):
        """Register a new face profile with associated embeddings"""
        pass
