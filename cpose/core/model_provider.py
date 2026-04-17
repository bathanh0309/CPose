import threading
from typing import Dict, Any
from ultralytics import YOLO

class ModelProvider:
    """
    Singleton class to manage and cache YOLO models.
    Ensures each model (by path) is loaded only once in memory.
    """
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelProvider, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_path: str) -> YOLO:
        """
        Returns a cached YOLO instance for the given path, 
        or loads it if not already present.
        """
        if model_path not in self._models:
            with self._lock:
                if model_path not in self._models:
                    print(f"[ModelProvider] Loading weight from: {model_path}")
                    self._models[model_path] = YOLO(model_path)
        return self._models[model_path]

    def clear(self):
        """Clears the model cache."""
        with self._lock:
            self._models.clear()

# Global accessor
def get_model(model_path: str) -> YOLO:
    return ModelProvider().get_model(model_path)
