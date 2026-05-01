from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class BaseVectorDB(ABC):
    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: List[int], metadata: Optional[List[dict]] = None):
        """Add embeddings with corresponding IDs to the database"""
        pass

    @abstractmethod
    def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k similar embeddings. Returns (distances, indices)"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Persist index to disk"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load index from disk"""
        pass
