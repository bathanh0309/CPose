from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

# 1. Define Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = DATA_DIR / "config"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
OUTPUT_DIR = DATA_DIR / "output_labels"
OUTPUT_POSE_DIR = DATA_DIR / "output_pose"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# 2. Init Environment & Directories
load_dotenv(BASE_DIR / ".env")

for runtime_dir in [CONFIG_DIR, RAW_VIDEOS_DIR, OUTPUT_DIR, OUTPUT_POSE_DIR, MODELS_DIR]:
    runtime_dir.mkdir(parents=True, exist_ok=True)

# 3. Config Setup
RESOURCES_FILE = CONFIG_DIR / "resources.txt"
UNIFIED_CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"

if not RESOURCES_FILE.exists():
    RESOURCES_FILE.write_text("# RTSP URLs, one per line\n", encoding="utf-8")

# Lazy-loaded singleton for config
_GLOBAL_CONFIG = None

def get_config():
    """Retrieve validated Pydantic config."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        from app.bootstrap.config_loader import load_config
        _GLOBAL_CONFIG = load_config(UNIFIED_CONFIG_FILE)
    return _GLOBAL_CONFIG

# 4. App Factory Proxy
def create_app():
    """Canonical entry point to create the Flask application."""
    from app.bootstrap.app_factory import create_app as _create_app
    
    config = get_config()
    return _create_app(static_dir=STATIC_DIR, config=config)

from app.bootstrap.app_factory import socketio

__all__ = ["create_app", "socketio", "get_config", "BASE_DIR", "STATIC_DIR"]
