"""
CPose — app/__init__.py
Flask application factory.
"""
import os
from pathlib import Path
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# ──────────────────────────────────────────
# Canonical paths (used project-wide)
# ──────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data"
CONFIG_DIR       = DATA_DIR / "config"
RAW_VIDEOS_DIR   = DATA_DIR / "raw_videos"
OUTPUT_DIR       = DATA_DIR / "output_labels"
MODELS_DIR       = BASE_DIR / "models"
STATIC_DIR       = BASE_DIR / "static"

RESOURCES_FILE   = CONFIG_DIR / "resources.txt"

# Model weights
MODEL_PHASE1     = MODELS_DIR / "yolov8n.pt"
MODEL_PHASE2     = MODELS_DIR / "yolov8l.pt"

# Defaults
DEFAULT_STORAGE_LIMIT_GB = 10.0

# Ensure runtime directories exist
for _d in [CONFIG_DIR, RAW_VIDEOS_DIR, OUTPUT_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────
# SocketIO instance (shared)
# ──────────────────────────────────────────
socketio = SocketIO()


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cpose-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload

    CORS(app)

    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    # Init SocketIO
    socketio.init_app(app, cors_allowed_origins="*", async_mode="eventlet")

    return app
