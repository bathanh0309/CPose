"""
app.py — Entry point
Chạy: python app.py

Cấu trúc:
    camera_web/
    ├── app.py          <- entry point (file này)
    ├── api.py          <- Flask routes / Blueprint
    ├── camera.py       <- CameraStream class
    ├── detection.py    <- YOLOv8, logger, config loader
    ├── cameras.json    <- danh sach camera (tu tao)
    ├── requirements.txt
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── css/style.css
    │   └── js/main.js
    └── logs/
        ├── detections.jsonl
        └── crops/
"""

import logging
import socket
from flask import Flask

from api       import api_bp
from camera    import cameras
from detection import get_model, autostart_cameras, LOG_FILE, CROP_DIR

# Flask app
app = Flask(__name__)
app.register_blueprint(api_bp)   # toan bo routes tu api.py

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Pre-loading YOLOv8n model...")
    get_model()

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("=" * 60)
    print("  RTSP Web Monitor + YOLOv8 Human Detection")
    print("=" * 60)
    print(f"  Local:   http://127.0.0.1:5000")
    print(f"  Network: http://{local_ip}:5000   <- chia se link nay")
    print(f"  Logs:    {LOG_FILE}")
    print(f"  Crops:   {CROP_DIR}")
    print("=" * 60)

    logging.info("Auto-starting cameras from cameras.json...")
    autostart_cameras(cameras)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)