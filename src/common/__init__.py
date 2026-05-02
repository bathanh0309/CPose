"""Shared utilities for CPose AI modules."""
from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class PersistenceManager:
    """Small compatibility persistence shim without a separate source file."""

    def __init__(self, persist_path: str, embedding_dim: int = 512) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._next_id = 1
        self._embeddings: dict[int, np.ndarray] = {}
        self._last_seen: dict[tuple[int, str], str] = {}

    def get_next_global_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def register_global_id(self, global_id: int, camera: str, embedding: np.ndarray, bbox: list[int] | None = None) -> None:
        self._embeddings[global_id] = np.asarray(embedding, dtype=np.float32)
        self._last_seen[(global_id, camera)] = datetime.now().isoformat()

    def get_embedding(self, global_id: int) -> np.ndarray | None:
        value = self._embeddings.get(global_id)
        return value.copy() if value is not None else None

    def get_last_seen(self, global_id: int, camera: str) -> str | None:
        return self._last_seen.get((global_id, camera))

    def get_statistics(self) -> dict[str, Any]:
        return {"total_global_ids": len(self._embeddings), "next_global_id": self._next_id}

    def close(self) -> None:
        return None


_persistence_module = types.ModuleType(__name__ + ".persistence")
_persistence_module.PersistenceManager = PersistenceManager
sys.modules.setdefault(__name__ + ".persistence", _persistence_module)


__all__ = ["PersistenceManager"]
