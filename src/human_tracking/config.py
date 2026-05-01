from __future__ import annotations

from pathlib import Path

from src.human_detection.config import resolve_detection_model


def resolve_tracking_model(model: str | Path | None = None) -> str:
    return resolve_detection_model(model)
