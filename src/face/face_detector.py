from __future__ import annotations

from typing import Any

import cv2


class OptionalFaceDetector:
    def __init__(self, min_face_size: int = 40) -> None:
        self.min_face_size = min_face_size
        self.available = False
        self._cascade: Any = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self._cascade = cascade
                self.available = True
        except Exception:
            self.available = False

    def detect(self, crop: Any) -> tuple[list[float] | None, float | None, str]:
        if crop is None or crop.size == 0:
            return None, None, "NO_FACE"
        if not self.available:
            return None, None, "MODEL_MISSING"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size))
        if len(faces) == 0:
            return None, None, "NO_FACE"
        x, y, w, h = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)[0]
        if w < self.min_face_size or h < self.min_face_size:
            return None, None, "FACE_TOO_SMALL"
        return [float(x), float(y), float(x + w), float(y + h)], min(1.0, (w * h) / max(crop.shape[0] * crop.shape[1], 1)), "OK"
