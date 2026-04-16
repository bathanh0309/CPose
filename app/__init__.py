"""Main Package Init. Exports paths and application factory."""

from __future__ import annotations
from pathlib import Path

# Extract Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = DATA_DIR / "config"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
OUTPUT_DIR = DATA_DIR / "output_labels"
OUTPUT_POSE_DIR = DATA_DIR / "output_pose"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

RESOURCES_FILE = CONFIG_DIR / "resources.txt"
UNIFIED_CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"
POSE_CONFIG_FILE = UNIFIED_CONFIG_FILE

MODEL_PHASE1 = MODELS_DIR / "yolov8n.pt"
MODEL_PHASE2 = MODELS_DIR / "yolo11n.pt"
MODEL_PHASE3 = MODELS_DIR / "yolo11n-pose.pt"

# Runtime directories initialization
for runtime_dir in [CONFIG_DIR, RAW_VIDEOS_DIR, OUTPUT_DIR, OUTPUT_POSE_DIR, MODELS_DIR]:
    runtime_dir.mkdir(parents=True, exist_ok=True)

if not RESOURCES_FILE.exists():
    RESOURCES_FILE.write_text("# RTSP URLs, one per line\n", encoding="utf-8")

# Import logic from bootstrap layer to avoid god-file anti-pattern
from app.bootstrap.app_factory import create_app as _create_app, socketio

# Single global reference to active config, initialized later if needed
_GLOBAL_CONFIG = None

def get_config():
    """Retrieve the global validated config, initializing it on first use."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        from app.bootstrap.config_loader import load_config
        _GLOBAL_CONFIG = load_config(UNIFIED_CONFIG_FILE)
    return _GLOBAL_CONFIG

def create_app():
    """Proxy app factory."""
    # We trigger config loading early to catch errors using Pydantic Validation
    _ = get_config()
    return _create_app(static_dir=STATIC_DIR)

__all__ = ["create_app", "socketio", "get_config"]
