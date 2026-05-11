from __future__ import annotations

import os
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

# ==== NEW: model & config paths / defaults ====
# Paths to model weights used by Phase 1/2/3 and pose config file.
# Defaults point at the checked-in `models/product` weights.
MODEL_PHASE1 = MODELS_DIR / "human_detect" / "yolov8n.pt"
MODEL_PHASE2 = MODELS_DIR / "human_detect" / "yolov8n.pt"
MODEL_PHASE3 = MODELS_DIR / "pose_estimation" / "yolov8n-pose.pt"

# Pose config: prefer a dedicated pose config in data/config, fall back to unified config elsewhere.
POSE_CONFIG_FILE = CONFIG_DIR / "pose_config.yaml"

# Default storage limit (GB) used by recording APIs if client omits value
DEFAULT_STORAGE_LIMIT_GB = 10.0
# ==============================================

# 2. Init Environment & Directories
load_dotenv(BASE_DIR / ".env")

for runtime_dir in [CONFIG_DIR, RAW_VIDEOS_DIR, OUTPUT_DIR, OUTPUT_POSE_DIR, MODELS_DIR]:
    runtime_dir.mkdir(parents=True, exist_ok=True)

# 3. Config Setup
RESOURCES_FILE = CONFIG_DIR / "resources.txt"
UNIFIED_CONFIG_FILE = CONFIGS_DIR / "unified_config.yaml"
APP_CONFIG_ENV_VAR = "CPOSE_APP_CONFIG"
DASHBOARD_URL_ENV_VAR = "CPOSE_DASHBOARD_URL"
CORS_ORIGINS_ENV_VAR = "CPOSE_CORS_ORIGINS"

if not RESOURCES_FILE.exists():
    RESOURCES_FILE.write_text("# RTSP URLs, one per line\n", encoding="utf-8")

# Lazy-loaded singleton for config
_GLOBAL_CONFIG = None


def _resolve_app_config_path() -> Path:
    raw_path = os.environ.get(APP_CONFIG_ENV_VAR, "").strip()
    if not raw_path:
        return UNIFIED_CONFIG_FILE
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def _apply_runtime_overrides(config):
    dashboard_url = os.environ.get(DASHBOARD_URL_ENV_VAR, "").strip()
    if dashboard_url:
        config.project.dashboard_url = dashboard_url

    cors_origins = os.environ.get(CORS_ORIGINS_ENV_VAR, "").strip()
    if cors_origins:
        config.server.cors_origins = [
            origin.strip()
            for origin in cors_origins.split(",")
            if origin.strip()
        ]
    return config


def get_config():
    """Retrieve validated Pydantic config."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        from app.bootstrap.config_loader import load_config

        _GLOBAL_CONFIG = _apply_runtime_overrides(load_config(_resolve_app_config_path()))
    return _GLOBAL_CONFIG


# 4. App Factory Proxy
def create_app():
    """Canonical entry point to create the Flask application."""
    from app.bootstrap.app_factory import create_app as _create_app

    config = get_config()
    return _create_app(static_dir=STATIC_DIR, config=config)


from app.bootstrap.app_factory import socketio

__all__ = [
    "create_app",
    "socketio",
    "get_config",
    "BASE_DIR",
    "STATIC_DIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "RAW_VIDEOS_DIR",
    "OUTPUT_DIR",
    "OUTPUT_POSE_DIR",
    "MODELS_DIR",
    "MODEL_PHASE1",
    "MODEL_PHASE2",
    "MODEL_PHASE3",
    "POSE_CONFIG_FILE",
    "DEFAULT_STORAGE_LIMIT_GB",
]
