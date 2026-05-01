from __future__ import annotations

from pathlib import Path
from typing import Any


class OptionalAntiSpoofing:
    def __init__(self, enabled: bool = False, model_path: str | Path | None = None) -> None:
        self.enabled = enabled
        self.model_path = Path(model_path) if model_path else None
        self.available = bool(enabled and self.model_path and self.model_path.exists())

    def check(self, _face_crop: Any) -> tuple[str, str]:
        if not self.enabled:
            return "unchecked", "ANTI_SPOOF_DISABLED"
        if not self.available:
            return "unchecked", "MODEL_MISSING"
        return "unchecked", "MODEL_MISSING"
