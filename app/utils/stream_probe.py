"""
CPose — app/utils/stream_probe.py
Probe an RTSP stream to obtain resolution, FPS, and available preset resolutions.
Uses OpenCV; falls back to a set of common resolutions that the camera may support.
"""
from __future__ import annotations

import logging

import cv2

logger = logging.getLogger("[StreamProbe]")

# Common IP camera resolutions to offer as options
COMMON_RESOLUTIONS = [
    (3840, 2160, "4K UHD"),
    (2560, 1440, "QHD 1440p"),
    (1920, 1080, "Full HD 1080p"),
    (1280,  720, "HD 720p"),
    ( 960,  540, "qHD 540p"),
    ( 640,  480, "VGA 480p"),
    ( 320,  240, "QVGA 240p"),
]


class StreamProber:
    """Probe RTSP streams for capabilities."""

    def probe(self, url: str, timeout_ms: int = 8000) -> dict:
        """
        Open an RTSP stream briefly to read its native resolution and FPS.

        Returns dict with:
            width, height, fps, resolutions (list of options ≤ native),
            or error (str) if connection fails.
        """
        logger.info("Probing stream: %s", url)
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            cap.release()
            return {"error": f"Cannot open stream: {url}"}

        native_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if native_w == 0 or native_h == 0:
            return {"error": "Stream opened but returned zero resolution — may need auth"}

        fps = round(native_fps, 2) if native_fps and native_fps > 0 else 25.0

        # Build resolution list: native + all common resolutions smaller than native
        native_mp = native_w * native_h
        resolutions = [{"width": native_w, "height": native_h, "label": f"Native {native_w}x{native_h}"}]
        for rw, rh, label in COMMON_RESOLUTIONS:
            if rw * rh < native_mp:
                resolutions.append({"width": rw, "height": rh, "label": label})

        logger.info(
            "Probe result: %dx%d @ %.2f fps  (%d resolution options)",
            native_w, native_h, fps, len(resolutions),
        )
        return {
            "width": native_w,
            "height": native_h,
            "fps": fps,
            "resolutions": resolutions,
        }
