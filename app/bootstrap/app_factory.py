import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# We'll initialize socketio here (unbound to app initially)
socketio = SocketIO()

def create_app(static_dir: Path) -> Flask:
    """Create and configure the CPose Flask application."""
    # Ensure current working directory has .env
    base_dir = static_dir.parent
    load_dotenv(base_dir / ".env", override=False)

    flask_app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static",
    )
    flask_app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cpose-dev-secret")
    flask_app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    CORS(flask_app)

    # Register blueprints safely across directory structures
    from app.api import api_bp
    flask_app.register_blueprint(api_bp)

    @flask_app.get("/")
    def dashboard():
        return send_from_directory(str(static_dir), "index.html")

    # Initialize Socket.IO with standard eventlet 
    socketio.init_app(flask_app, cors_allowed_origins="*", async_mode="eventlet")

    # Register websockets after socketio init
    try:
        import app.api.ws_handlers  # noqa: F401
    except Exception:
        import logging
        logging.getLogger("[App]").exception("Failed to import ws_handlers")

    return flask_app
