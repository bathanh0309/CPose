"""
CPose — app/api/routes.py
All REST API endpoints and SocketIO event handlers.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_from_directory

import app as _app_module
from app.services.phase1_recorder import RecorderManager
from app.services.phase2_analyzer import Analyzer
from app.utils.file_handler import FileHandler
from app.utils.stream_probe import StreamProber

logger = logging.getLogger("[API]")

api_bp = Blueprint("api", __name__, url_prefix="/api")

# ── Singletons ──────────────────────────────────────────────────────────────
_recorder  = RecorderManager()
_analyzer  = Analyzer()
_fh        = FileHandler()
_prober    = StreamProber()


# ═══════════════════════════════════════════════════════════════════════════
#  Serve static dashboard
# ═══════════════════════════════════════════════════════════════════════════
@api_bp.route("/", defaults={"path": ""})
@api_bp.route("/<path:path>", endpoint="catch_all")
def serve_dashboard(path: str):
    """Serve the single-page dashboard."""
    static_dir = Path(current_app.static_folder)
    if path and (static_dir / path).exists():
        return send_from_directory(str(static_dir), path)
    return send_from_directory(str(static_dir), "index.html")


# ═══════════════════════════════════════════════════════════════════════════
#  Config endpoints
# ═══════════════════════════════════════════════════════════════════════════

@api_bp.route("/config/upload", methods=["POST"])
def upload_config():
    """Upload resources.txt containing RTSP URLs."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files accepted"}), 400

    dest = _app_module.RESOURCES_FILE
    f.save(str(dest))

    cameras = _fh.parse_resources(dest)
    logger.info("Config uploaded — %d cameras found", len(cameras))
    return jsonify({"message": "Config saved", "cameras": cameras})


@api_bp.route("/config/cameras", methods=["GET"])
def get_cameras():
    """Return list of cameras defined in resources.txt."""
    if not _app_module.RESOURCES_FILE.exists():
        return jsonify({"cameras": []})
    cameras = _fh.parse_resources(_app_module.RESOURCES_FILE)
    return jsonify({"cameras": cameras})


# ═══════════════════════════════════════════════════════════════════════════
#  Stream probe
# ═══════════════════════════════════════════════════════════════════════════

@api_bp.route("/cameras/probe", methods=["POST"])
def probe_camera():
    """
    Probe an RTSP stream to obtain resolution and FPS.
    Body: { "url": "rtsp://...", "cam_id": "01" }
    """
    body = request.get_json(force=True, silent=True) or {}
    url = body.get("url", "").strip()
    cam_id = body.get("cam_id", "00")

    if not url:
        return jsonify({"error": "url required"}), 400

    info = _prober.probe(url)
    if info.get("error"):
        return jsonify({"error": info["error"], "cam_id": cam_id}), 502

    return jsonify({
        "cam_id": cam_id,
        "url": url,
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "resolutions": info["resolutions"],
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — Recording control
# ═══════════════════════════════════════════════════════════════════════════

@api_bp.route("/recording/start", methods=["POST"])
def start_recording():
    """
    Start Phase 1 recording on selected cameras.
    Body: {
      "cameras": [{"cam_id": "01", "url": "rtsp://...", "width": 1920, "height": 1080}],
      "storage_limit_gb": 10.0
    }
    """
    body = request.get_json(force=True, silent=True) or {}
    cameras = body.get("cameras", [])
    storage_limit_gb = float(body.get("storage_limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB))

    if not cameras:
        return jsonify({"error": "No cameras specified"}), 400

    if _recorder.is_running():
        return jsonify({"error": "Recording already running"}), 409

    _recorder.start(cameras, storage_limit_gb, _app_module.RAW_VIDEOS_DIR, _app_module.MODEL_PHASE1)
    return jsonify({"message": "Recording started", "cameras": len(cameras)})


@api_bp.route("/recording/stop", methods=["POST"])
def stop_recording():
    """Stop Phase 1 recording."""
    _recorder.stop()
    return jsonify({"message": "Recording stopped"})


@api_bp.route("/recording/status", methods=["GET"])
def recording_status():
    """Return current recording status."""
    return jsonify(_recorder.status())


# ═══════════════════════════════════════════════════════════════════════════
#  Video management
# ═══════════════════════════════════════════════════════════════════════════

@api_bp.route("/videos", methods=["GET"])
def list_videos():
    """List all recorded clips in raw_videos."""
    videos = _fh.list_videos(_app_module.RAW_VIDEOS_DIR)
    return jsonify({"videos": videos})


@api_bp.route("/videos/<filename>", methods=["DELETE"])
def delete_video(filename: str):
    """Delete a specific clip."""
    target = _app_module.RAW_VIDEOS_DIR / filename
    if not target.exists() or not filename.endswith(".mp4"):
        return jsonify({"error": "File not found"}), 404
    target.unlink()
    return jsonify({"message": f"{filename} deleted"})


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — Analysis control
# ═══════════════════════════════════════════════════════════════════════════

@api_bp.route("/analysis/start", methods=["POST"])
def start_analysis():
    """
    Start Phase 2 analysis on a folder of clips.
    Body: { "video_dir": "data/raw_videos" }
    """
    body = request.get_json(force=True, silent=True) or {}
    video_dir = Path(body.get("video_dir", str(_app_module.RAW_VIDEOS_DIR)))

    if not video_dir.is_dir():
        return jsonify({"error": f"Directory not found: {video_dir}"}), 400

    if _analyzer.is_running():
        return jsonify({"error": "Analysis already running"}), 409

    clips = list(video_dir.glob("*.mp4"))
    if not clips:
        return jsonify({"error": "No .mp4 files found in directory"}), 400

    _analyzer.start(clips, _app_module.OUTPUT_DIR, _app_module.MODEL_PHASE2)
    return jsonify({"message": "Analysis started", "clips": len(clips)})


@api_bp.route("/analysis/stop", methods=["POST"])
def stop_analysis():
    """Interrupt running Phase 2 analysis."""
    _analyzer.stop()
    return jsonify({"message": "Analysis stopped"})


@api_bp.route("/analysis/status", methods=["GET"])
def analysis_status():
    """Return Phase 2 status."""
    return jsonify(_analyzer.status())


@api_bp.route("/analysis/results", methods=["GET"])
def analysis_results():
    """List all output results (PNG + TXT pairs)."""
    results = _fh.list_results(_app_module.OUTPUT_DIR)
    return jsonify({"results": results})


# ═══════════════════════════════════════════════════════════════════════════
#  Storage management
# ═══════════════════════════════════════════════════════════════════════════

@api_bp.route("/storage/info", methods=["GET"])
def storage_info():
    """Return storage usage of raw_videos."""
    info = _fh.storage_info(_app_module.RAW_VIDEOS_DIR)
    return jsonify(info)


@api_bp.route("/storage/limit", methods=["POST"])
def set_storage_limit():
    """
    Update the storage limit enforced during recording.
    Body: { "limit_gb": 10.0 }
    """
    body = request.get_json(force=True, silent=True) or {}
    limit_gb = float(body.get("limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB))
    _recorder.set_storage_limit(limit_gb)
    return jsonify({"message": f"Limit set to {limit_gb} GB"})
