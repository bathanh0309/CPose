"""
Shared settings for migrated reference modules.

These defaults exist to keep `app.detectors.*` importable after moving files
from repo root into the `app/` package. Adjust as you integrate the modules.
"""

from __future__ import annotations

from pathlib import Path

from app.utils.runtime_config import get_runtime_section

BASE_DIR = Path(__file__).resolve().parent.parent

_DETECTOR_CFG = get_runtime_section("detectors")


def _resolve_config_path(raw_path: str, fallback_relative: str) -> str:
    value = str(raw_path or "").strip() or fallback_relative
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return str(candidate)


# Body detector (Ultralytics YOLO)
YOLO_PATH = _resolve_config_path(_DETECTOR_CFG.get("yolo_path", "models/yolov8n.pt"), "models/yolov8n.pt")

# Face recognition (InsightFace) - embeddings folder
FACE_DATA_PATH = _resolve_config_path(_DETECTOR_CFG.get("face_data_path", "data/face_db"), "data/face_db")
FACE_MIN_SIZE = int(_DETECTOR_CFG.get("face_min_size", 50))
