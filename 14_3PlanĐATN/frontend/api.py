"""
api.py — Flask routes
Import và đăng ký vào app từ app.py: app.register_blueprint(api_bp)
"""

import cv2
import sys
import json
from flask import Blueprint, Response, render_template, jsonify, request, send_from_directory

from camera import cameras, CameraStream
from detection import load_camera_config
from detection import LOG_FILE, CROP_DIR

api_bp = Blueprint("api", __name__)


# ─── Pages ───────────────────────────────────────────────────────────────────
@api_bp.route("/")
def index():
    return render_template("index.html")


# ─── MJPEG stream ─────────────────────────────────────────────────────────────
def _mjpeg_generator(cam: CameraStream, my_stream_id: int):
    import time
    import numpy as np
    BOUNDARY    = b"--frame\r\n"
    HEADER      = b"Content-Type: image/jpeg\r\n\r\n"
    TAIL        = b"\r\n"
    placeholder = _make_placeholder_jpeg(cam.cam_id)

    while cam.stream_id == my_stream_id:
        frame = cam.get_jpeg() or placeholder
        yield BOUNDARY + HEADER + frame + TAIL
        time.sleep(1 / 30)


def _make_placeholder_jpeg(cam_id: int) -> bytes:
    import numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (13, 17, 23)
    for x in range(0, 640, 40):
        cv2.line(img, (x, 0), (x, 360), (21, 38, 45), 1)
    for y in range(0, 360, 40):
        cv2.line(img, (0, y), (640, y), (21, 38, 45), 1)
    cv2.putText(img, f"CAM-0{cam_id + 1}  NO SIGNAL",
                (200, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (72, 79, 88), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()


@api_bp.route("/video/<int:cam_id>")
def video_feed(cam_id: int):
    if cam_id < 0 or cam_id >= 4:
        return "Invalid camera", 404
    cam = cameras[cam_id]
    return Response(
        _mjpeg_generator(cam, cam.stream_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─── Camera control ───────────────────────────────────────────────────────────
@api_bp.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)
    for i, entry in enumerate(data[:4]):
        src = entry.get("source")
        if src is None or src == "":
            cameras[i].stop()
        else:
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            cameras[i].start(src)
    return jsonify({"ok": True})


@api_bp.route("/api/stop", methods=["POST"])
def api_stop():
    for c in cameras:
        c.stop()
    return jsonify({"ok": True})


# ─── Status ───────────────────────────────────────────────────────────────────
@api_bp.route("/api/status")
def api_status():
    return jsonify([
        {
            "id":           c.cam_id,
            "status":       c.status,
            "info":         c.info,
            "fps":          c.fps_real,
            "person_count": c.person_count,
            "detection":    c.detection_enabled,
        }
        for c in cameras
    ])


# ─── Detection toggle ─────────────────────────────────────────────────────────
@api_bp.route("/api/detection_toggle", methods=["POST"])
def api_detection_toggle():
    data    = request.get_json(force=True)
    cam_id  = int(data.get("cam_id", 0))
    enabled = bool(data.get("enabled", True))
    if 0 <= cam_id < 4:
        cameras[cam_id].detection_enabled = enabled
        if not enabled:
            cameras[cam_id].person_count = 0
    return jsonify({"ok": True, "cam_id": cam_id, "enabled": enabled})


# ─── Logs ─────────────────────────────────────────────────────────────────────
@api_bp.route("/api/logs")
def api_logs():
    limit = int(request.args.get("limit", 50))
    if not LOG_FILE.exists():
        return jsonify([])
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return jsonify(list(reversed(entries)))


@api_bp.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    if LOG_FILE.exists():
        LOG_FILE.write_text("")
    return jsonify({"ok": True})


@api_bp.route("/crops/<path:filename>")
def serve_crop(filename):
    return send_from_directory(str(CROP_DIR), filename)


# ─── Config ───────────────────────────────────────────────────────────────────
@api_bp.route("/api/camera_config")
def api_camera_config():
    return jsonify(load_camera_config())
