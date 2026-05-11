"""Face recognition, alignment, and anti-spoofing facade."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.face_antispoof import crop, infer, load_model, preprocess, preprocess_batch, process_with_logits
from models.face_detect.face_detect import FaceRecognition


@dataclass(slots=True)
class FaceRecognizer:
    """Small facade around RetinaFace/ArcFace plus MiniFASNet anti-spoofing."""

    gallery_dir: str | Path
    liveness_model: str | Path | None = None
    threshold: float = 0.4
    spoof_threshold: float = 0.6

    def __post_init__(self) -> None:
        self.recognizer = FaceRecognition(threshold=self.threshold, data_path=Path(self.gallery_dir))
        self.recognizer.load_face_database()
        self.session: Any | None = None
        self.input_name: str | None = None
        if self.liveness_model:
            self.session, self.input_name = load_model(self.liveness_model)

    def detect(self, frame: Any) -> list[dict]:
        return self.recognizer.face_detect(frame) or []

    def match(self, embedding: Any) -> tuple[str | None, float]:
        return self.recognizer.match_face(embedding)

    def is_real_face(self, face_crop: Any) -> dict[str, Any]:
        if self.session is None or self.input_name is None:
            return {"is_real": None, "prob_real": None, "reason": "ANTI_SPOOF_MODEL_NOT_LOADED"}
        preds = infer([face_crop], self.session, self.input_name, 128)
        return process_with_logits(preds[0], self.spoof_threshold)


__all__ = [
    "FaceRecognition",
    "FaceRecognizer",
    "crop",
    "infer",
    "load_model",
    "preprocess",
    "preprocess_batch",
    "process_with_logits",
]
