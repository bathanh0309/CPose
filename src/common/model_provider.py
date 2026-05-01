from __future__ import annotations

import threading
from typing import Any


class ModelProvider:
    """Thread-safe in-process cache for Ultralytics YOLO models."""

    _instance = None
    _lock = threading.Lock()
    _models: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_yolo(self, model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required. Install requirements.txt") from exc

        if model_path not in self._models:
            with self._lock:
                if model_path not in self._models:
                    print(f"[ModelProvider] Loading model: {model_path}")
                    self._models[model_path] = YOLO(model_path)
        return self._models[model_path]

    def clear(self) -> None:
        with self._lock:
            self._models.clear()


def get_yolo_model(model_path: str) -> Any:
    return ModelProvider().get_yolo(model_path)

