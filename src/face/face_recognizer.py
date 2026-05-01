from __future__ import annotations

from typing import Any


class OptionalFaceRecognizer:
    def __init__(self, enabled: bool = False) -> None:
        self.available = False
        if enabled:
            try:
                import insightface  # noqa: F401

                self.available = False
            except Exception:
                self.available = False

    def embed(self, _face_crop: Any) -> tuple[list[float] | None, int | None, str]:
        if not self.available:
            return None, None, "MODEL_MISSING"
        return None, None, "MODEL_MISSING"
