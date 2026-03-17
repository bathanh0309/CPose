"""
detection.py — YOLOv8 model loader, detection logger, camera config loader
"""

import cv2
import json
import logging
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
LOG_DIR   = BASE_DIR / "logs"
CROP_DIR  = LOG_DIR / "crops"
LOG_FILE  = LOG_DIR / "detections.jsonl"
LOG_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

CAMERA_CONFIG_FILE = BASE_DIR / "cameras.json"

# ─── YOLOv8 ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _yolo_available = True
except ImportError:
    _yolo_available = False
    logging.warning("ultralytics not installed — detection disabled. pip install ultralytics")

_model_lock = threading.Lock()
_model      = None


def get_model():
    global _model
    if not _yolo_available:
        return None
    with _model_lock:
        if _model is None:
            try:
                _model = YOLO("D:\ModA_FaceReg\models\yolov8n.pt")
                logging.info("YOLOv8n model loaded.")
            except Exception as e:
                logging.error(f"Failed to load YOLOv8: {e}")
                _model = None
    return _model

# ─── Detection logger ─────────────────────────────────────────────────────────
_log_lock = threading.Lock()


def log_detection(cam_id: int, cam_label: str, count: int, crops: list[np.ndarray]) -> dict:
    """Ghi log JSONL + lưu ảnh crop. Trả về entry dict."""
    ts      = datetime.now()
    ts_str  = ts.strftime("%Y%m%d_%H%M%S")
    ts_iso  = ts.isoformat(timespec="seconds")

    crop_files = []
    for idx, crop in enumerate(crops):
        fname = f"{ts_str}_cam{cam_id}_p{idx}.jpg"
        cv2.imwrite(str(CROP_DIR / fname), crop)
        crop_files.append(fname)

    entry = {
        "time":      ts_iso,
        "cam_id":    cam_id,
        "cam_label": cam_label,
        "count":     count,
        "crops":     crop_files,
    }
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


# ─── Camera config ────────────────────────────────────────────────────────────
def load_camera_config() -> list[dict]:
    """Đọc cameras.json, trả về list config."""
    if not CAMERA_CONFIG_FILE.exists():
        return []
    try:
        with open(CAMERA_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Cannot load cameras.json: {e}")
        return []


def autostart_cameras(cameras_list) -> None:
    """
    Nhận danh sách CameraStream và start từng camera theo cameras.json.
    Tách tham số để tránh circular import.
    """
    config = load_camera_config()
    for entry in config:
        cam_id = int(entry.get("id", -1))
        source = entry.get("source", "")
        if cam_id < 0 or cam_id >= len(cameras_list):
            continue
        if source == "" or source is None:
            continue
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        logging.info(f"Auto-start CAM-{cam_id} → {source}")
        cameras_list[cam_id].start(source)