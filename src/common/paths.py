from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "dataset"
DATA = DATA_DIR
MULTICAM_DIR = PROJECT_ROOT / "data-test"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUTS = OUTPUT_DIR
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ANNOTATIONS = ANNOTATIONS_DIR
MODELS_DIR = PROJECT_ROOT / "models"
MODELS = MODELS_DIR


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
