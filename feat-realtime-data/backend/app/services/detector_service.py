# Dịch vụ phát hiện người bằng YOLO và fallback MOG2 khi cần.
"""
Detector service for person-only detection using shared YOLO weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_SHARED_YOLO_NANO = _REPO_ROOT / "models" / "yolov8n.pt"
_LOCAL_YOLO_NANO = _BACKEND_ROOT / "yolov8n.pt"

PERSON_CLASS = 0


class DetectorService:
    def __init__(self) -> None:
        self._model = None
        self._mode: str = "none"
        self._bg_subtractor: Optional[cv2.BackgroundSubtractor] = None
        self._det_w = _settings.detection_width
        self._det_h = _settings.detection_height
        self._conf = _settings.detection_conf

    def load(self) -> None:
        """
        Load local YOLO nano weights. Fall back to MOG2 only when ultralytics
        is unavailable in the environment.
        """
        try:
            from ultralytics import YOLO  # type: ignore

            if _settings.detector_model == "yolo_nano":
                preferred_weights = _SHARED_YOLO_NANO if _SHARED_YOLO_NANO.exists() else _LOCAL_YOLO_NANO
                if not preferred_weights.exists():
                    raise FileNotFoundError(
                        f"YOLO weights not found in shared or local paths: "
                        f"{_SHARED_YOLO_NANO} | {_LOCAL_YOLO_NANO}"
                    )
                weights = str(preferred_weights)
            else:
                weights = "yolov8s.pt"

            self._model = YOLO(weights)
            self._model.overrides["classes"] = [PERSON_CLASS]
            self._mode = "yolo"
            logger.info("Detector loaded: YOLO (%s) on %s", weights, _settings.detector_device)
        except ImportError:
            logger.warning("ultralytics not installed; falling back to MOG2 motion detector.")
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=50,
                detectShadows=False,
            )
            self._mode = "mog2"
        except Exception as exc:
            logger.error("Detector failed to load local YOLO weights: %s", exc)
            self._mode = "none"

    def detect(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (self._det_w, self._det_h))

        if self._mode == "yolo" and self._model is not None:
            return self._yolo_detect(small)
        if self._mode == "mog2" and self._bg_subtractor is not None:
            return self._mog2_detect(small)
        return False

    def _yolo_detect(self, frame: np.ndarray) -> bool:
        results = self._model.predict(
            frame,
            device=_settings.detector_device,
            conf=self._conf,
            verbose=False,
            stream=False,
        )
        for result in results:
            if result.boxes and len(result.boxes) > 0:
                return True
        return False

    def _mog2_detect(self, frame: np.ndarray) -> bool:
        mask = self._bg_subtractor.apply(frame)
        motion_pixels = cv2.countNonZero(mask)
        threshold = (self._det_w * self._det_h) * 0.01
        return motion_pixels > threshold

    @property
    def mode(self) -> str:
        return self._mode


detector_service = DetectorService()
