import logging
import os
import socket
from pathlib import Path
from ipaddress import ip_address

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from app.utils.config_schema import AppConfig
from .logging_setup import setup_logging

logger = logging.getLogger("[AppFactory]")
# Keep Socket.IO/Engine.IO logs quiet in product mode to avoid noisy connection-abort traces on Windows.
socketio = SocketIO(logger=False, engineio_logger=False)

_DEFAULT_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:3000",
]


def _iter_local_ip_origins(port: int) -> list[str]:
    origins: list[str] = []
    seen: set[str] = set()

    def _add(ip: str) -> None:
        try:
            parsed = ip_address(ip)
        except ValueError:
            return
        if parsed.version != 4 or parsed.is_loopback:
            return
        value = f"http://{parsed}:{port}"
        if value in seen:
            return
        seen.add(value)
        origins.append(value)

    hostnames = {
        socket.gethostname(),
        socket.getfqdn(),
    }
    for hostname in hostnames:
        try:
            _, _, addresses = socket.gethostbyname_ex(hostname)
        except OSError:
            continue
        for address in addresses:
            _add(address)

    return origins


def _resolve_allowed_origins(config: AppConfig) -> list[str]:
    origins: list[str] = []
    seen: set[str] = set()

    def _add(origin: str | None) -> None:
        if not origin:
            return
        normalized = str(origin).strip().rstrip("/")
        if not normalized or normalized == "*" or normalized in seen:
            return
        seen.add(normalized)
        origins.append(normalized)

    for origin in _DEFAULT_ORIGINS:
        _add(origin)

    for origin in _iter_local_ip_origins(config.server.port):
        _add(origin)

    _add(getattr(config.project, "dashboard_url", ""))

    raw_origins = getattr(config.server, "cors_origins", "*")
    if isinstance(raw_origins, str):
        if raw_origins.strip() != "*":
            for origin in raw_origins.split(","):
                _add(origin)
    else:
        for origin in raw_origins:
            _add(origin)

    return origins or list(_DEFAULT_ORIGINS)


def create_app(static_dir: Path, config: AppConfig) -> Flask:
    """
    Application Factory: Initializes Flask, SocketIO and Core Services.
    """
    setup_logging(config.logging)
    logger.info("Initializing CPose application factory...")

    flask_app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static",
    )

    flask_app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cpose-dev-secret")
    flask_app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

    allowed_origins = _resolve_allowed_origins(config)
    CORS(
        flask_app,
        resources={
            r"/*": {
                "origins": allowed_origins,
                "supports_credentials": True,
            }
        },
    )

    from app.core.recognizer_service import RecognizerService
    from app.services.recorder import RecorderManager
    from app.services.registration_service import RegistrationManager

    recorder = RecorderManager()
    registration = RegistrationManager()

    def _socket_emit(event: str, data):
        socketio.emit(event, data)

    recognizer = RecognizerService(
        socket_callback=_socket_emit,
        registration_callback=None,
        pose_model_path=Path(config.models.pose_model_path),
    )
    recognizer.start_worker()

    flask_app.config["recorder"] = recorder
    flask_app.config["recognizer"] = recognizer
    flask_app.config["registration"] = registration
    flask_app.config["app_config"] = config
    flask_app.config["on_socket_emit_cb"] = _socket_emit

    def on_clip_ready(clip_path: Path, cam_id: str):
        logger.info("Auto-processing trigger: %s from %s", clip_path.name, cam_id)
        recognizer.enqueue_clip(cam_id, Path(clip_path))

    flask_app.config["on_clip_ready_cb"] = on_clip_ready

    _register_blueprints(flask_app)
    _register_ui_routes(flask_app, static_dir)

    socketio.init_app(
        flask_app,
        cors_allowed_origins=allowed_origins,
        async_mode=config.server.socket_async_mode,
        manage_session=False,
    )

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
