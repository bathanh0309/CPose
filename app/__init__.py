"""Flask application factory and canonical project paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

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
UNIFIED_CONFIG_FILE = CONFIGS_DIR / "config.yaml"
POSE_CONFIG_FILE = UNIFIED_CONFIG_FILE

MODEL_PHASE1 = MODELS_DIR / "yolov8n.pt"
MODEL_PHASE2 = MODELS_DIR / "yolo11n.pt"
MODEL_PHASE3 = MODELS_DIR / "yolo11n-pose.pt"

DEFAULT_STORAGE_LIMIT_GB = 10.0

for runtime_dir in [CONFIG_DIR, RAW_VIDEOS_DIR, OUTPUT_DIR, OUTPUT_POSE_DIR, MODELS_DIR]:
    runtime_dir.mkdir(parents=True, exist_ok=True)

if not RESOURCES_FILE.exists():
    RESOURCES_FILE.write_text("# RTSP URLs, one per line\n", encoding="utf-8")

socketio = SocketIO()


def create_app() -> Flask:
    """Create and configure the CPose Flask application."""
    load_dotenv(BASE_DIR / ".env", override=False)

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cpose-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    CORS(app)

    from app.api import api_bp

    app.register_blueprint(api_bp)

    @app.get("/")
    def dashboard():
        return send_from_directory(str(STATIC_DIR), "index.html")

    socketio.init_app(app, cors_allowed_origins="*", async_mode="eventlet")
    return app
