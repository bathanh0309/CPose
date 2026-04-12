"""All Flask REST endpoints for the canonical CPose app."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

import app as _app_module
from app.services.phase1_recorder import RecorderManager
from app.services.phase2_analyzer import Analyzer
from app.services.phase3_recognizer import PoseADLRecognizer
from app.utils.file_handler import StorageManager
from app.utils.stream_probe import StreamProber

logger = logging.getLogger("[API]")

api_bp = Blueprint("api", __name__, url_prefix="/api")

_recorder = RecorderManager()
_analyzer = Analyzer()
_pose = PoseADLRecognizer()
_storage = StorageManager()
_prober = StreamProber()


def _resolve_dir(raw_value: str | None, default: Path) -> Path:
    if not raw_value:
        return default
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else (_app_module.BASE_DIR / candidate)


@api_bp.route("/config/upload", methods=["POST"])
def upload_config():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    upload = request.files["file"]
    filename = upload.filename or ""
    if not filename.lower().endswith(".txt"):
        return jsonify({"error": "Only .txt files are supported"}), 400

    upload.save(str(_app_module.RESOURCES_FILE))
    cameras = _storage.parse_resources(_app_module.RESOURCES_FILE)
    logger.info("Loaded %d cameras from resources.txt", len(cameras))
    return jsonify({"message": "resources.txt saved", "cameras": cameras})


@api_bp.route("/config/cameras", methods=["GET"])
def get_cameras():
    if not _app_module.RESOURCES_FILE.exists():
        return jsonify({"cameras": []})
    return jsonify({"cameras": _storage.parse_resources(_app_module.RESOURCES_FILE)})


@api_bp.route("/cameras/probe", methods=["POST"])
def probe_camera():
    body = request.get_json(force=True, silent=True) or {}
    url = str(body.get("url", "")).strip()
    cam_id = str(body.get("cam_id", "00")).zfill(2)
    if not url:
        return jsonify({"error": "url is required"}), 400

    info = _prober.probe(url)
    if info.get("error"):
        return jsonify({"error": info["error"], "cam_id": cam_id}), 502

    return jsonify(
        {
            "cam_id": cam_id,
            "url": url,
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "resolutions": info["resolutions"],
        }
    )


@api_bp.route("/recording/start", methods=["POST"])
def start_recording():
    body = request.get_json(force=True, silent=True) or {}
    cameras = body.get("cameras") or []
    try:
        storage_limit_gb = float(body.get("storage_limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB))
    except (TypeError, ValueError):
        return jsonify({"error": "storage_limit_gb must be numeric"}), 400

    if not cameras:
        return jsonify({"error": "No cameras specified"}), 400
    if _recorder.is_running():
        return jsonify({"error": "Recording is already running"}), 409

    _recorder.start(
        cameras=cameras,
        storage_limit_gb=storage_limit_gb,
        output_dir=_app_module.RAW_VIDEOS_DIR,
        model_path=_app_module.MODEL_PHASE1,
    )
    return jsonify({"message": "Recording started", "cameras": len(cameras)})


@api_bp.route("/recording/stop", methods=["POST"])
def stop_recording():
    _recorder.stop()
    return jsonify({"message": "Recording stopped"})


@api_bp.route("/recording/status", methods=["GET"])
def recording_status():
    return jsonify(_recorder.status())


@api_bp.route("/videos", methods=["GET"])
def list_videos():
    return jsonify({"videos": _storage.list_videos(_app_module.RAW_VIDEOS_DIR)})


@api_bp.route("/videos/<path:filename>", methods=["DELETE"])
def delete_video(filename: str):
    if not filename.endswith(".mp4"):
        return jsonify({"error": "Only .mp4 files can be deleted"}), 400

    target = _app_module.RAW_VIDEOS_DIR / filename
    if not target.exists() or not target.is_file():
        return jsonify({"error": "File not found"}), 404

    target.unlink()
    return jsonify({"message": f"{filename} deleted"})


@api_bp.route("/analysis/start", methods=["POST"])
def start_analysis():
    body = request.get_json(force=True, silent=True) or {}
    video_dir = _resolve_dir(body.get("video_dir") or body.get("folder"), _app_module.RAW_VIDEOS_DIR)
    if not video_dir.is_dir():
        return jsonify({"error": f"Directory not found: {video_dir}"}), 400
    if _analyzer.is_running():
        return jsonify({"error": "Analysis is already running"}), 409

    clips = sorted(video_dir.glob("*.mp4"))
    if not clips:
        return jsonify({"error": "No .mp4 files found in the selected folder"}), 400

    _analyzer.start(clips, _app_module.OUTPUT_DIR, _app_module.MODEL_PHASE2)
    return jsonify({"message": "Analysis started", "clips": len(clips)})


@api_bp.route("/analysis/stop", methods=["POST"])
def stop_analysis():
    _analyzer.stop()
    return jsonify({"message": "Analysis stop requested"})


@api_bp.route("/analysis/status", methods=["GET"])
def analysis_status():
    return jsonify(_analyzer.status())


@api_bp.route("/analysis/results", methods=["GET"])
def analysis_results():
    folder = _resolve_dir(request.args.get("folder"), _app_module.OUTPUT_DIR)
    if not folder.exists():
        return jsonify({"results": []})
    return jsonify({"results": _storage.list_results(folder)})


@api_bp.route("/pose/start", methods=["POST"])
def start_pose():
    body = request.get_json(force=True, silent=True) or {}
    video_dir = _resolve_dir(body.get("folder"), _app_module.RAW_VIDEOS_DIR)
    save_overlay = bool(body.get("save_overlay", True))

    if not video_dir.is_dir():
        return jsonify({"error": f"Directory not found: {video_dir}"}), 400
    if _pose.is_running():
        return jsonify({"error": "Pose analysis is already running"}), 409

    clips = sorted(video_dir.glob("*.mp4"))
    if not clips:
        return jsonify({"error": "No .mp4 files found in the selected folder"}), 400

    _pose.start(
        clips=clips,
        output_dir=_app_module.OUTPUT_POSE_DIR,
        model_path=_app_module.MODEL_PHASE3,
        config_path=_app_module.POSE_CONFIG_FILE,
        save_overlay=save_overlay,
    )
    return jsonify({"status": "started", "total_clips": len(clips), "save_overlay": save_overlay})


@api_bp.route("/pose/stop", methods=["POST"])
def stop_pose():
    _pose.stop()
    return jsonify({"status": "stop_requested"})


@api_bp.route("/pose/status", methods=["GET"])
def pose_status():
    return jsonify(_pose.status())


@api_bp.route("/pose/results", methods=["GET"])
def pose_results():
    folder = _resolve_dir(request.args.get("folder"), _app_module.OUTPUT_POSE_DIR)
    if not folder.exists():
        return jsonify({"results": []})
    return jsonify({"results": _storage.list_pose_results(folder)})


@api_bp.route("/pose/adl_summary", methods=["GET"])
def pose_adl_summary():
    clip_stem = str(request.args.get("clip", "")).strip()
    if not clip_stem:
        return jsonify({"error": "clip is required"}), 400

    folder = _resolve_dir(request.args.get("folder"), _app_module.OUTPUT_POSE_DIR)
    summary = _storage.get_pose_summary(folder, clip_stem)
    if summary is None:
        return jsonify({"error": f"Pose result not found for clip: {clip_stem}"}), 404
    return jsonify({"clip": clip_stem, "adl_distribution": summary})


@api_bp.route("/storage/info", methods=["GET"])
def storage_info():
    return jsonify(_storage.storage_info(_app_module.RAW_VIDEOS_DIR))


@api_bp.route("/storage/limit", methods=["POST"])
def set_storage_limit():
    body = request.get_json(force=True, silent=True) or {}
    try:
        limit_gb = float(body.get("limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB))
    except (TypeError, ValueError):
        return jsonify({"error": "limit_gb must be numeric"}), 400

    if limit_gb <= 0:
        return jsonify({"error": "limit_gb must be greater than zero"}), 400

    _recorder.set_storage_limit(limit_gb)
    return jsonify({"message": f"Storage limit updated to {limit_gb:.2f} GB"})
