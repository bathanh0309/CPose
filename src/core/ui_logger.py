from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any


VALID_LEVELS = {"INFO", "WARNING", "ERROR", "METRIC"}


class UILogger:
    def __init__(self, max_lines: int = 300):
        self.max_lines = int(max_lines)
        self._logs: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_lines))
        self._metrics: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_lines))

    def log(self, camera_id: str, level: str, module: str, message: str, data: dict[str, Any] | None = None):
        level = level.upper()
        if level not in VALID_LEVELS:
            level = "INFO"
        item = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "camera_id": str(camera_id),
            "level": level,
            "module": str(module),
            "message": str(message),
            "data": data or {},
        }
        self._logs[str(camera_id)].append(item)
        return item

    def metric(self, camera_id: str, metrics_dict: dict[str, Any]):
        payload = dict(metrics_dict)
        payload.setdefault("camera_id", str(camera_id))
        payload.setdefault("time", datetime.now().strftime("%H:%M:%S"))
        self._metrics[str(camera_id)].append(payload)
        self.log(str(camera_id), "METRIC", str(payload.get("module", "Metrics")), str(payload.get("message", "OK")), payload)
        return payload

    def get_logs(self, camera_id: str):
        return list(self._logs[str(camera_id)])

    def get_metrics(self, camera_id: str):
        return list(self._metrics[str(camera_id)])

    def status(self):
        camera_ids = sorted(set(self._logs) | set(self._metrics))
        return {
            camera_id: {
                "logs": len(self._logs[camera_id]),
                "metrics": len(self._metrics[camera_id]),
                "last_log": self._logs[camera_id][-1] if self._logs[camera_id] else None,
                "last_metric": self._metrics[camera_id][-1] if self._metrics[camera_id] else None,
            }
            for camera_id in camera_ids
        }


ui_logger = UILogger()
