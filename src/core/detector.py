"""Person and face detection entrypoints used by runtime pipelines."""
from __future__ import annotations

from models.face_detect.face_detect import FaceRecognition
from models.human_detect.detector import PersonDetector, reset_detection_counter, resolve_detection_model
from models.human_detect.human_detect import PersonDetector as SimplePersonDetector

__all__ = [
    "FaceRecognition",
    "PersonDetector",
    "SimplePersonDetector",
    "reset_detection_counter",
    "resolve_detection_model",
]
