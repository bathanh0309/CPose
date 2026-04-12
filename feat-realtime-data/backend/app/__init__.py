# Khởi tạo package backend và giảm log nhiễu của OpenCV/FFmpeg.
"""
Application package bootstrap.

Set OpenCV / FFmpeg logging controls before any submodule imports `cv2`.
This suppresses noisy decoder warnings such as:
  [h264 @ ...] SEI type ... truncated at ...
"""

from __future__ import annotations

import os


# Silence OpenCV's own logger and lower the FFmpeg bridge log level to errors only.
# FFmpeg log levels come from AV_LOG_*; 16 == error, -8 == quiet.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


def _silence_opencv_runtime_logs() -> None:
    """Best-effort runtime log suppression for OpenCV Python builds."""
    try:
        import cv2  # type: ignore

        if hasattr(cv2, "setLogLevel"):
            try:
                cv2.setLogLevel(2)  # ERROR
            except Exception:
                pass

        utils = getattr(cv2, "utils", None)
        logging_mod = getattr(utils, "logging", None) if utils else None
        if logging_mod and hasattr(logging_mod, "setLogLevel"):
            log_level_error = getattr(logging_mod, "LOG_LEVEL_ERROR", None)
            if log_level_error is not None:
                logging_mod.setLogLevel(log_level_error)
    except Exception:
        # Startup should not fail if a particular OpenCV build lacks logging hooks.
        pass


_silence_opencv_runtime_logs()
