"""All Flask REST endpoints for the canonical CPose app."""

from __future__ import annotations
import logging
from pathlib import Path
from flask import Blueprint, Response, jsonify, request, current_app
from werkzeug.utils import secure_filename

import app as _app_module
from app.services.recorder import RecorderManager
from app.services.analyzer import Analyzer
from app.core.recognizer_service import RecognizerService
from app.utils.file_handler import StorageManager
from app.utils.stream_probe import StreamProber
from app.utils.file_handler import sort_multicam_clips
from app.services.registration_service import RegistrationManager

logger = logging.getLogger("[API]")

# Blueprint for all APIs
api_bp = Blueprint("api", __name__, url_prefix="/api")

# Helpers to get services from app config
def get_recorder() -> RecorderManager:
    return current_app.config["recorder"]

def get_pose() -> RecognizerService:
    return current_app.config["recognizer"]

_analyzer = Analyzer()
_storage = StorageManager()
_prober = StreamProber()
_registration = RegistrationManager()

# --- Internal Helpers ---
def _body_json() -> dict:
    return request.get_json(force=True, silent=True) or {}

def _ok(**kwargs):
    return jsonify({"ok": True, **kwargs})

def _error(message: str, code: int = 400, **kwargs):
    return jsonify({"ok": False, "error": message, **kwargs}), code

def _resolve_dir(raw_value: str | None, default: Path) -> Path:
    if not raw_value:
        return default
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else (_app_module.BASE_DIR / candidate)

def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

# ===================================================================
#                          CONFIG & CAMERA
# ===================================================================

@api_bp.route("/config/upload", methods=["POST"])
def upload_config():
    if "file" not in request.files:
        return _error("No file provided", 400)

    upload = request.files["file"]
    filename = upload.filename or ""

    if not filename.lower().endswith(".txt"):
        return _error("Only .txt files are supported", 400)

    upload.save(str(_app_module.RESOURCES_FILE))
    cameras = _storage.parse_resources(_app_module.RESOURCES_FILE)

    from app.api.ws_handlers import emit_event_log
    emit_event_log(f"Loaded resources.txt ({len(cameras)} cameras)", "CFG")

    return _ok(message="resources.txt saved", cameras=cameras)

@api_bp.route("/config/cameras", methods=["GET"])
def get_cameras():
    if not _app_module.RESOURCES_FILE.exists():
        return jsonify({"cameras": []})
    return jsonify({"cameras": _storage.parse_resources(_app_module.RESOURCES_FILE)})

@api_bp.route("/cameras/<cam_id>/snapshot", methods=["GET"])
def camera_snapshot(cam_id: str):
    jpeg = get_recorder().get_snapshot(cam_id)
    if jpeg is None:
        return Response(status=204)

    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )

# ===================================================================
#                          RECORDING (PHASE 1)
# ===================================================================

@api_bp.route("/recording/start", methods=["POST"])
def start_recording():
    body = _body_json()
    selected_cams = body.get("cameras") or []

    if isinstance(selected_cams, str):
        selected_cams = [selected_cams]

    all_cams = _storage.parse_resources(_app_module.RESOURCES_FILE)
    cameras = []

    if len(selected_cams) > 0 and isinstance(selected_cams[0], str):
        for c in all_cams:
            if c.get("cam_id") in selected_cams or c.get("label") in selected_cams:
                cameras.append(c)
    else:
        cameras = selected_cams

    if not cameras:
        return _error("No valid cameras found or specified", 400)

    if get_recorder().is_running():
        return _error("Recording is already running", 409)

    try:
        storage_limit_gb = float(body.get("storage_limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB))
    except (TypeError, ValueError):
        return _error("storage_limit_gb must be numeric", 400)

    get_recorder().start(
        cameras=cameras,
        storage_limit_gb=storage_limit_gb,
        output_dir=_app_module.RAW_VIDEOS_DIR,
        model_path=_app_module.MODEL_PHASE1,
        config=body.get("phase1_config", {}),
        on_clip_ready=current_app.config.get("on_clip_ready_cb"),
    )

    from app.api.ws_handlers import emit_event_log, emit_rec_status
    emit_event_log("Recorder started", "SYS")
    emit_rec_status(is_recording=True)

    return _ok(message="Recording started", cameras=len(cameras))

@api_bp.route("/recording/stop", methods=["POST"])
def stop_recording():
    get_recorder().stop()
    from app.api.ws_handlers import emit_event_log, emit_rec_status
    emit_event_log("Recorder stopped", "SYS")
    emit_rec_status(is_recording=False)
    return _ok(message="Recording stopped")

@api_bp.route("/recording/status", methods=["GET"])
def recording_status():
    return jsonify(get_recorder().status())

# ===================================================================
#                          POSE & ADL (PHASE 3)
# ===================================================================

@api_bp.route("/pose/start", methods=["POST"])
def start_pose():
    body = _body_json()
    mode = str(body.get("mode", "multicam_folder")).strip().lower()
    recognizer = get_pose()

    if mode == "rtsp":
        cameras_req = body.get("cameras") or []
        all_cams = _storage.parse_resources(_app_module.RESOURCES_FILE)
        selected_cams = [c for c in all_cams if c.get("cam_id") in cameras_req]
        if not selected_cams:
            return _error("No valid cameras for RTSP mode", 400)

        if not get_recorder().is_running():
            get_recorder().start(
                cameras=selected_cams,
                storage_limit_gb=float(body.get("storage_limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB)),
                output_dir=_app_module.RAW_VIDEOS_DIR,
                model_path=_app_module.MODEL_PHASE1,
                on_clip_ready=current_app.config.get("on_clip_ready_cb"),
            )

        from app.api.ws_handlers import emit_event_log
        emit_event_log("Pose pipeline started in RTSP mode", "SYS")
        return _ok(status="started", mode="rtsp", cameras=len(selected_cams))

    if mode == "multicam_folder":
        folder = body.get("folder")
        if folder:
            video_dir = _resolve_dir(folder, _app_module.RAW_VIDEOS_DIR)
        else:
            video_dir = _app_module.BASE_DIR / "data" / "multicam"

        if not video_dir.is_dir():
            return _error(f"Directory not found: {video_dir}", 400)

        clips = sort_multicam_clips(video_dir.glob("*.mp4"))
        if not clips:
            return _error("No .mp4 files found in the selected folder", 400)

        for clip in clips:
            cam_id = _app_module.extract_multicam_camera_id(clip) if hasattr(_app_module, "extract_multicam_camera_id") else "unknown"
            recognizer.enqueue_clip(cam_id, clip)

        from app.api.ws_handlers import emit_event_log
        emit_event_log(f"Pose pipeline started in multicam_folder mode ({len(clips)} clips)", "SYS")
        return _ok(status="started", mode="multicam_folder", total_clips=len(clips))

    return _error(f"Unsupported mode: {mode}", 400)

@api_bp.route("/pose/stop", methods=["POST"])
def stop_pose():
    get_pose().stop()
    get_recorder().stop()
    from app.api.ws_handlers import emit_event_log, emit_rec_status
    emit_event_log("Pose processing stopped", "SYS")
    emit_rec_status(is_recording=False)
    return _ok(status="stopped")

@api_bp.route("/pose/status", methods=["GET"])
def pose_status():
    status = _json_safe(get_pose().get_status())
    lamp_state = status.get("lamp_state") or {
        "cam01": "IDLE", "cam02": "IDLE", "cam03": "IDLE", "cam04": "IDLE",
    }
    normalized = {
        "running": status.get("running", False),
        "mode": status.get("mode", "idle"),
        "current_clip": status.get("current_clip"),
        "current_cam": status.get("current_cam"),
        "current_frame": status.get("current_frame", 0),
        "total_frames": status.get("total_frames", 0),
        "fps": status.get("fps", 0),
        "conf": status.get("conf", 0),
        "adl": status.get("adl", "unknown"),
        "lamp_state": lamp_state,
        "pending_results": status.get("pending_results", []),
    }
    return jsonify(normalized)

@api_bp.route("/pose/snapshot/<view>", methods=["GET"])
def pose_snapshot(view: str):
    if view not in ("original", "processed"):
        return _error("Invalid view type", 400)

    jpeg = get_pose().get_snapshot(view)
    if jpeg is None:
        return Response(status=204)

    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )

@api_bp.route("/pose/pending_results", methods=["GET"])
def pose_pending_results():
    pending = _json_safe(get_pose().pending_results())
    return jsonify({"pending_results": pending})

@api_bp.route("/pose/save_result", methods=["POST"])
def save_pose_result():
    body = _body_json()
    clip_stem = body.get("clip_stem")
    if not clip_stem:
        return _error("clip_stem is required", 400)
    
    result = get_pose().save_pending_result(clip_stem)
    if "error" in result:
        return _error(result["error"], 400)
    
    return _ok(message=f"Result saved for {clip_stem}", **result)

# ===================================================================
#                          WORKSPACE HELPERS
# ===================================================================

@api_bp.route("/workspace/load_multicam_default", methods=["POST"])
def load_multicam_default():
    video_dir = _app_module.BASE_DIR / "data" / "multicam"
    if not video_dir.exists():
        return _error(f"Directory not found: {video_dir}", 404)

    clips = sort_multicam_clips(video_dir.glob("*.mp4"))
    if not clips:
        return _error("No .mp4 files found in data/multicam", 400)

    items = []
    for clip in clips:
        cam_id = _app_module.extract_multicam_camera_id(clip) if hasattr(_app_module, "extract_multicam_camera_id") else "unknown"
        items.append({
            "name": clip.name,
            "cam": cam_id,
            "path": str(clip),
            "url": f"/api/video/{clip.relative_to(_app_module.DATA_DIR).as_posix()}",
        })

    from app.api.ws_handlers import emit_event_log
    emit_event_log(f"Loaded default multicam folder ({len(items)} clips)", "MC")
    return _ok(mode="multicam_folder", clips=items, total=len(items))

@api_bp.route("/workspace/start_multicam", methods=["POST"])
def workspace_start_multicam():
    body = _body_json()
    folder = body.get("folder")
    if not folder:
        folder = str((_app_module.BASE_DIR / "data" / "multicam").resolve())

    # We can call start_pose internal logic directly or via request context
    with current_app.test_request_context(
        "/api/pose/start",
        method="POST",
        json={"mode": "multicam_folder", "folder": folder},
    ):
        return start_pose()

@api_bp.route("/workspace/stop", methods=["POST"])
def workspace_stop():
    get_pose().stop()
    get_recorder().stop()
    from app.api.ws_handlers import emit_event_log, emit_rec_status
    emit_event_log("Workspace stopped", "SYS")
    emit_rec_status(is_recording=False)
    return _ok(status="stopped")

# ===================================================================
#                          REGISTRATION
# ===================================================================

@api_bp.route("/registration/next_id", methods=["GET"])
def get_next_registration_id():
    next_id = _registration.get_next_id()
    return jsonify({"next_id": next_id})

@api_bp.route("/registration/start", methods=["POST"])
def start_registration():
    body = _body_json()
    source = body.get("source")
    rtsp_url = body.get("rtsp_url")
    user_name = body.get("name")
    user_age = body.get("age", "??")
    person_id = body.get("person_id") or _registration.get_next_id()

    if not user_name:
        return _error("Name is required", 400)

    if source == "rtsp" and not rtsp_url:
        return _error("RTSP URL is required for RTSP source", 400)

    camera_source = "local" if source == "local" else rtsp_url

    def on_progress(data):
        from app.api.ws_handlers import emit_metric_log
        # Standard socket notify
        current_app.config["on_socket_emit_cb"]("registration_progress", data)
        emit_metric_log(cam="REG", event=data.get("message", "registration_progress"))

    def on_done(data):
        current_app.config["on_socket_emit_cb"]("registration_done", data)
        if data.get("status") == "success":
            get_pose().refresh_face_database()

    session_id = _registration.start_session(
        source=camera_source,
        user_name=user_name,
        user_age=user_age,
        person_id=person_id,
        on_progress=on_progress,
        on_done=on_done,
    )

    return _ok(session_id=session_id, status="started", user_id=person_id, person_id=person_id)

@api_bp.route("/registration/stop", methods=["POST"])
def stop_registration():
    body = _body_json()
    session_id = body.get("session_id")
    if not session_id:
        return _error("session_id is required", 400)
    _registration.stop_session(session_id)
    return _ok(message="Stopped")

@api_bp.route("/registration/snapshot/<session_id>", methods=["GET"])
def registration_snapshot(session_id: str):
    jpeg = _registration.get_snapshot(session_id)
    if jpeg is None:
        return Response(status=204)

    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )

# --- Serving Static Videos ---
@api_bp.route("/video/<path:filepath>")
def serve_video(filepath):
    data_dir = _app_module.BASE_DIR / "data"
    from flask import send_from_directory
    return send_from_directory(str(data_dir), filepath)

# --- Helper for direct socket notify ---
def _on_socket_emit(event, data):
    from app import socketio
    socketio.emit(event, data)

# Link callback in create_app for registration
# Callback setup moved to factory or handled via direct imports
def setup_api_callbacks(app: Flask):
    app.config["on_socket_emit_cb"] = _on_socket_emit
