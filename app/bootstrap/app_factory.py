import os
import logging
from pathlib import Path
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from app.utils.config_schema import AppConfig
from .logging_setup import setup_logging

logger = logging.getLogger("[AppFactory]")
socketio = SocketIO()

def create_app(static_dir: Path, config: AppConfig) -> Flask:
    """
    Application Factory: Initializes Flask, SocketIO and Core Services.
    """
    # Initialize logging first so all subsequent logs follow the config
    setup_logging(config.logging)
    logger.info("Initializing CPose application factory...")

    # 1. Base Framework Setup
    flask_app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static",
    )
    
    flask_app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cpose-dev-secret")
    
    # Increase limit to avoid failure for video/folder uploads
    flask_app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB
    
    CORS(flask_app, resources={r"/*": {"origins": config.server.cors_origins}})

    # 2. Service Orchestration (Producer-Consumer)
    from app.services.recorder import RecorderManager
    from app.services.recognizer.orchestrator import RecognizerService
    
    # Init Consumer
    recognizer = RecognizerService(config, socketio)
    recognizer.start()
    
    # Init Producer
    recorder = RecorderManager()
    
    # We store them in app.config for the API to access
    flask_app.config["recorder"] = recorder
    flask_app.config["recognizer"] = recognizer

    # Callback to link Producer -> Consumer
    def on_clip_ready(clip_path: Path, cam_id: str):
        logger.info(f"Auto-processing trigger: {clip_path.name} from {cam_id}")
        recognizer.enqueue_clip(cam_id, Path(clip_path))

    flask_app.config["on_clip_ready_cb"] = on_clip_ready

    # 3. Protocol & Routing Registration
    _register_blueprints(flask_app)
    
    from app.api.routes import setup_api_callbacks
    setup_api_callbacks(flask_app)
    
    _register_ui_routes(flask_app, static_dir)
    
    # 4. Initialize SocketIO with chosen async mode
    socketio.init_app(
        flask_app, 
        cors_allowed_origins=config.server.cors_origins, 
        async_mode=config.server.socket_async_mode
    )
    
    # Register handlers after init
    _register_socket_handlers()

    return flask_app

def _register_blueprints(app: Flask):
    from app.api import api_bp
    app.register_blueprint(api_bp)

def _register_ui_routes(app: Flask, static_dir: Path):
    @app.get("/")
    def dashboard():
        return send_from_directory(str(static_dir), "index.html")

def _register_socket_handlers():
    """Import handlers to register them with the socketio instance."""
    try:
        import app.api.ws_handlers  # noqa: F401
    except Exception:
        logger.exception("Failed to register SocketIO handlers.")
