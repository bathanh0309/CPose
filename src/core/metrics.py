from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleMetrics:
    camera_id: str
    module: str
    frame_idx: int
    fps: float
    device: str
    status: str = "running"
    message: str = "OK"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "camera_id": self.camera_id,
            "module": self.module,
            "frame_idx": int(self.frame_idx),
            "fps": float(self.fps),
            "device": self.device,
            "status": self.status,
            "message": self.message,
        }
        payload.update(self.extra)
        return payload
