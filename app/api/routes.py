"""All Flask REST endpoints for the canonical CPose app."""

from __future__ import annotations
import logging
from pathlib import Path
from flask import Blueprint, Response, jsonify, request

import app as _app_module
from app.services.phase1_recorder import RecorderManager
from app.services.phase2_analyzer import Analyzer
from app.services.phase3_recognizer import PoseADLRecognizer
from app.utils.file_handler import StorageManager
from app.utils.stream_probe import StreamProber
from app.utils.file_handler import sort_multicam_clips

logger = logging.getLogger("[API]")

# Tạo Blueprint cho tất cả các API
api_bp = Blueprint("api", __name__, url_prefix="/api")

# Khởi tạo các service chính (singleton style)
_recorder = RecorderManager()
_analyzer = Analyzer()
_pose = PoseADLRecognizer()
_storage = StorageManager()
_prober = StreamProber()


def _resolve_dir(raw_value: str | None, default: Path) -> Path:
    """Hàm hỗ trợ chuyển đổi đường dẫn tương đối thành tuyệt đối."""
    if not raw_value:
        return default
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else (_app_module.BASE_DIR / candidate)


# ===================================================================
#                          CONFIG & CAMERA
# ===================================================================

@api_bp.route("/config/upload", methods=["POST"])
def upload_config():
    """
    Upload file resources.txt chứa thông tin cấu hình camera.
    File sẽ được lưu vào đường dẫn RESOURCES_FILE và sau đó được parse để lấy danh sách camera.
    """
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
    """
    Lấy danh sách tất cả các camera đã được cấu hình trong file resources.txt.
    Nếu file chưa tồn tại, trả về danh sách rỗng.
    """
    if not _app_module.RESOURCES_FILE.exists():
        return jsonify({"cameras": []})

    return jsonify({"cameras": _storage.parse_resources(_app_module.RESOURCES_FILE)})


@api_bp.route("/config/load_local", methods=["POST"])
def load_local_test_cameras():
    """Tạo resources.txt từ tất cả các file .mp4 trong thư mục data/multicam để test Phase 1"""
    video_dir = _app_module.BASE_DIR / "data" / "multicam"
    if not video_dir.exists():
        return jsonify({"error": "Thư mục data/multicam không tồn tại"}), 404
        
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        return jsonify({"error": "Không có video .mp4 nào trong data/multicam"}), 404
    
    with open(_app_module.RESOURCES_FILE, "w", encoding="utf-8") as f:
        f.write("# Automatically generated for local testing\n")
        # Ensure sequential order cam01, cam02 etc.
        for i, path in enumerate(sorted(videos)):
            cam_str = f"cam{i+1:02d}"
            f.write(f"{cam_str} __ {str(path.resolve())}\n")
            
    cameras = _storage.parse_resources(_app_module.RESOURCES_FILE)
    logger.info("Loaded %d cameras dynamically from data/multicam", len(cameras))
    return jsonify({"message": f"Loaded {len(videos)} local videos as cameras", "cameras": cameras})


@api_bp.route("/cameras/probe", methods=["POST"])
def probe_camera():
    """
    Kiểm tra (probe) một camera stream (RTSP/HTTP...) để lấy thông tin kỹ thuật:
    độ phân giải, FPS, các resolution hỗ trợ.
    Dùng để test xem camera có hoạt động tốt trước khi ghi hình.
    """
    body = request.get_json(force=True, silent=True) or {}
    url = str(body.get("url", "")).strip()
    cam_id = str(body.get("cam_id", "00")).zfill(2)

    if not url:
        return jsonify({"error": "url is required"}), 400

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


# ===================================================================
#                   CAMERA SNAPSHOT & LIVE STATUS
# ===================================================================

@api_bp.route("/cameras/<cam_id>/snapshot", methods=["GET"])
def camera_snapshot(cam_id: str):
    """
    Trả về ảnh JPEG mới nhất (snapshot) của camera đang chạy dưới dạng image/jpeg.
    Nếu camera chưa chạy hoặc chưa có frame nào, trả về HTTP 204 No Content.
    Dùng để hiển thị live view trên giao diện web.
    """
    jpeg = _recorder.get_snapshot(cam_id)
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


@api_bp.route("/cameras/status", methods=["GET"])
def cameras_live_status():
    """
    Trả về trạng thái live của tất cả các camera đang chạy.
    """
    return jsonify({"cameras": _recorder.camera_status_list()})


# ===================================================================
#                       RECORDING (Phase 1)
# ===================================================================

@api_bp.route("/recording/start", methods=["POST"])
def start_recording():
    """
    Bắt đầu quá trình ghi hình (Phase 1).
    Cho phép chỉ định danh sách camera, giới hạn dung lượng lưu trữ,
    và các cấu hình bổ sung cho Phase 1.
    """
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

    # Phase 1 config overrides from body
    phase1_cfg = body.get("phase1_config", {})

    _recorder.start(
        cameras=cameras,
        storage_limit_gb=storage_limit_gb,
        output_dir=_app_module.RAW_VIDEOS_DIR,
        model_path=_app_module.MODEL_PHASE1,
        config=phase1_cfg,
    )

    return jsonify({"message": "Recording started", "cameras": len(cameras)})


@api_bp.route("/recording/stop", methods=["POST"])
def stop_recording():
    """
    Dừng quá trình ghi hình đang chạy (Phase 1).
    """
    _recorder.stop()
    return jsonify({"message": "Recording stopped"})


@api_bp.route("/recording/status", methods=["GET"])
def recording_status():
    """
    Lấy trạng thái hiện tại của quá trình ghi hình.
    """
    return jsonify(_recorder.status())


# ===================================================================
#                          VIDEOS MANAGEMENT
# ===================================================================

@api_bp.route("/videos", methods=["GET"])
def list_videos():
    """
    Liệt kê tất cả các file video (.mp4) đã ghi trong thư mục RAW_VIDEOS_DIR.
    """
    return jsonify({"videos": _storage.list_videos(_app_module.RAW_VIDEOS_DIR)})


@api_bp.route("/videos/<path:filename>", methods=["DELETE"])
def delete_video(filename: str):
    """
    Xóa một file video theo tên file.
    Chỉ cho phép xóa file có đuôi .mp4.
    """
    if not filename.endswith(".mp4"):
        return jsonify({"error": "Only .mp4 files can be deleted"}), 400

    target = _app_module.RAW_VIDEOS_DIR / filename
    if not target.exists() or not target.is_file():
        return jsonify({"error": "File not found"}), 404

    target.unlink()
    return jsonify({"message": f"{filename} deleted"})


# ===================================================================
#                    ANALYSIS (Phase 2)
# ===================================================================

@api_bp.route("/analysis/start", methods=["POST"])
def start_analysis():
    """
    Bắt đầu quá trình phân tích video (Phase 2).
    Nhận vào thư mục chứa các file video cần phân tích.
    """
    body = request.get_json(force=True, silent=True) or {}
    video_dir = _resolve_dir(body.get("video_dir") or body.get("folder"), _app_module.RAW_VIDEOS_DIR)

    if not video_dir.is_dir():
        return jsonify({"error": f"Directory not found: {video_dir}"}), 400

    if _analyzer.is_running():
        return jsonify({"error": "Analysis is already running"}), 409

    clips = sort_multicam_clips(video_dir.glob("*.mp4"))
    if not clips:
        return jsonify({"error": "No .mp4 files found in the selected folder"}), 400

    _analyzer.start(clips, _app_module.OUTPUT_DIR, _app_module.MODEL_PHASE2)

    return jsonify({"message": "Analysis started", "clips": len(clips)})


@api_bp.route("/analysis/stop", methods=["POST"])
def stop_analysis():
    """
    Yêu cầu dừng quá trình phân tích video đang chạy (Phase 2).
    """
    _analyzer.stop()
    return jsonify({"message": "Analysis stop requested"})


@api_bp.route("/analysis/status", methods=["GET"])
def analysis_status():
    """
    Lấy trạng thái hiện tại của quá trình phân tích (Phase 2).
    """
    return jsonify(_analyzer.status())


@api_bp.route("/analysis/results", methods=["GET"])
def analysis_results():
    """
    Liệt kê các kết quả phân tích trong thư mục chỉ định (mặc định là OUTPUT_DIR).
    """
    folder = _resolve_dir(request.args.get("folder"), _app_module.OUTPUT_DIR)
    if not folder.exists():
        return jsonify({"results": []})
    return jsonify({"results": _storage.list_results(folder)})


# ===================================================================
#                   POSE & ADL (Phase 3)
# ===================================================================

@api_bp.route("/pose/start", methods=["POST"])
def start_pose():
    """
    Bắt đầu quá trình nhận diện Pose và ADL (Phase 3).
    Có thể chọn thư mục video đầu vào và bật/tắt việc lưu video overlay.
    """
    body = request.get_json(force=True, silent=True) or {}
    video_dir = _resolve_dir(body.get("folder"), _app_module.RAW_VIDEOS_DIR)
    save_overlay = bool(body.get("save_overlay", True))

    if not video_dir.is_dir():
        return jsonify({"error": f"Directory not found: {video_dir}"}), 400

    if _pose.is_running():
        return jsonify({"error": "Pose analysis is already running"}), 409

    clips = sort_multicam_clips(video_dir.glob("*.mp4"))
    if not clips:
        return jsonify({"error": "No .mp4 files found in the selected folder"}), 400

    _pose.start(
        clips=clips,
        output_dir=_app_module.OUTPUT_POSE_DIR,
        model_path=_app_module.MODEL_PHASE3,
        config_path=_app_module.POSE_CONFIG_FILE,
        save_overlay=save_overlay,
    )
    pose_state = _pose.status()

    return jsonify({
        "status": "started",
        "total_clips": len(clips),
        "save_overlay": save_overlay,
        "lamp_state": pose_state.get("lamp_state", {}),
        "clip_queue": pose_state.get("clip_queue", []),
        "active_camera": pose_state.get("active_camera", ""),
    })


@api_bp.route("/pose/stop", methods=["POST"])
def stop_pose():
    """
    Yêu cầu dừng quá trình nhận diện Pose và ADL đang chạy (Phase 3).
    """
    _pose.stop()
    return jsonify({"status": "stop_requested"})


@api_bp.route("/pose/status", methods=["GET"])
def pose_status():
    """
    Lấy trạng thái hiện tại của quá trình Pose & ADL recognition (Phase 3).
    """
    return jsonify(_pose.status())


@api_bp.route("/pose/results", methods=["GET"])
def pose_results():
    """
    Liệt kê các kết quả Pose trong thư mục chỉ định (mặc định là OUTPUT_POSE_DIR).
    """
    folder = _resolve_dir(request.args.get("folder"), _app_module.OUTPUT_POSE_DIR)
    if not folder.exists():
        return jsonify({"results": []})
    return jsonify({"results": _storage.list_pose_results(folder)})


@api_bp.route("/pose/snapshot/<view>", methods=["GET"])
def pose_snapshot(view: str):
    """
    Trả về ảnh JPEG mới nhất (snapshot) của quá trình Pose đang chạy.
    view: 'original' hoặc 'processed'.
    """
    if view not in ("original", "processed"):
        return jsonify({"error": "Invalid view type"}), 400

    jpeg = _pose.get_snapshot(view)
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


@api_bp.route("/pose/adl_summary", methods=["GET"])
def pose_adl_summary():
    """
    Lấy tóm tắt phân bố các hoạt động ADL (Activities of Daily Living)
    của một clip video cụ thể (ví dụ: ngồi, đứng, nằm, đi lại...).
    """
    clip_stem = str(request.args.get("clip", "")).strip()
    if not clip_stem:
        return jsonify({"error": "clip is required"}), 400

    folder = _resolve_dir(request.args.get("folder"), _app_module.OUTPUT_POSE_DIR)
    summary = _storage.get_pose_summary(folder, clip_stem)

    if summary is None:
        return jsonify({"error": f"Pose result not found for clip: {clip_stem}"}), 404

    return jsonify({"clip": clip_stem, "adl_distribution": summary})


# ===================================================================
#                       STORAGE MANAGEMENT
# ===================================================================

@api_bp.route("/storage/info", methods=["GET"])
def storage_info():
    """
    Lấy thông tin dung lượng lưu trữ của thư mục video
    (dung lượng đã dùng, còn trống, số lượng file...).
    """
    return jsonify(_storage.storage_info(_app_module.RAW_VIDEOS_DIR))


@api_bp.route("/storage/limit", methods=["POST"])
def set_storage_limit():
    """
    Cập nhật giới hạn dung lượng lưu trữ tối đa cho quá trình ghi hình.
    """
    body = request.get_json(force=True, silent=True) or {}

    try:
        limit_gb = float(body.get("limit_gb", _app_module.DEFAULT_STORAGE_LIMIT_GB))
    except (TypeError, ValueError):
        return jsonify({"error": "limit_gb must be numeric"}), 400

    if limit_gb <= 0:
        return jsonify({"error": "limit_gb must be greater than zero"}), 400

    _recorder.set_storage_limit(limit_gb)
    return jsonify({"message": f"Storage limit updated to {limit_gb:.2f} GB"})


@api_bp.route("/video/<path:filepath>")
def serve_video(filepath):
    from flask import send_from_directory
    data_dir = _app_module.BASE_DIR / "data"
    return send_from_directory(data_dir, filepath)
