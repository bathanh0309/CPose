from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class PersonGateDetector:
    """
    Lightweight person detector for AI gating.

    This detector only answers:
        Is there a person in the frame?

    It also returns bbox + confidence so the UI can show person-gate detection.
    """

    def __init__(
        self,
        weights: str,
        fallback_weights: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: list[int] | None = None,
        min_box_area: float = 600,
        device: str | None = None,
    ):
        self.weights = Path(weights)
        self.fallback_weights = Path(fallback_weights) if fallback_weights else None

        if not self.weights.exists():
            if self.fallback_weights and self.fallback_weights.exists():
                self.weights = self.fallback_weights
            else:
                raise FileNotFoundError(
                    f"Person gate model not found: {weights}. "
                    f"Fallback not found: {fallback_weights}"
                )

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = classes if classes is not None else [0]
        self.min_box_area = float(min_box_area)
        self.device = device

        print(
            f"[PersonGate] model={self.weights} "
            f"device={self.device} conf={self.conf} imgsz={self.imgsz}"
        )

        self.model = YOLO(str(self.weights))

    def detect(self, frame: np.ndarray) -> tuple[bool, list[dict[str, Any]]]:
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        detections: list[dict[str, Any]] = []

        if result.boxes is None or len(result.boxes) == 0:
            return False, detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            bbox = xyxy[i].tolist()
            area = bbox_area(bbox)

            if area < self.min_box_area:
                continue

            detections.append({
                "bbox": bbox,
                "score": float(confs[i]),
                "class_id": int(cls_ids[i]),
                "area": float(area),
            })

        return len(detections) > 0, detections

    @staticmethod
    def draw_gate_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> None:
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            score = float(det.get("score", 0.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

            label = f"person {score:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class PersonGateDetector:
    """
    Lightweight person detector used only as a gate.

    It answers:
        Is there any person in the frame?

    It should not run ReID, ADL, PoseC3D, or heavy module logic.
    """

    def __init__(
        self,
        weights: str,
        fallback_weights: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: list[int] | None = None,
        min_box_area: float = 800,
        device: str | None = None,
    ):
        self.weights = Path(weights)
        self.fallback_weights = Path(fallback_weights) if fallback_weights else None

        if not self.weights.exists():
            if self.fallback_weights and self.fallback_weights.exists():
                print(f"[PersonGate] model not found: {self.weights}; using fallback={self.fallback_weights}")
                self.weights = self.fallback_weights
            else:
                raise FileNotFoundError(
                    f"Person gate model not found: {weights}. "
                    f"Fallback also not found: {fallback_weights}"
                )

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = classes if classes is not None else [0]
        self.min_box_area = float(min_box_area)
        self.device = self._normalize_device(self.weights, device)

        print(
            f"[PersonGate] loading model={self.weights} "
            f"device={self.device} conf={self.conf} imgsz={self.imgsz}"
        )

        self.model = YOLO(str(self.weights))

    @staticmethod
    def _is_openvino_model(weights: Path) -> bool:
        text = str(weights).lower()
        return "openvino_model" in text or weights.suffix.lower() == ".xml"

    @classmethod
    def _normalize_device(cls, weights: Path, device: str | None) -> str | None:
        if not device:
            return None

        device_text = str(device).strip()
        if cls._is_openvino_model(weights):
            if device_text.lower().startswith("intel:"):
                return device_text
            return f"intel:{device_text.lower()}"

        if device_text.lower().startswith("intel:"):
            return "cpu"
        return device_text

    @staticmethod
    def _is_openvino_gpu_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in ("igc_check", "cisa", "gpu compiler", "cldnn", "openvino"))

    def _reload(self, weights: Path, device: str | None, reason: str) -> None:
        self.weights = weights
        self.device = self._normalize_device(weights, device)
        print(f"[PersonGate] {reason}; loading model={self.weights} device={self.device}")
        self.model = YOLO(str(self.weights))

    def _predict(self, frame: np.ndarray):
        return self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

    def detect(self, frame: np.ndarray) -> tuple[bool, list[dict[str, Any]]]:
        try:
            results = self._predict(frame)
        except Exception as exc:
            if (
                self._is_openvino_gpu_error(exc)
                and self._is_openvino_model(self.weights)
                and str(self.device).lower() == "intel:gpu"
            ):
                self._reload(self.weights, "intel:cpu", f"OpenVINO GPU failed ({type(exc).__name__}: {exc})")
                results = self._predict(frame)
            elif self.fallback_weights and self.fallback_weights.exists() and self.weights != self.fallback_weights:
                self._reload(self.fallback_weights, "cpu", f"primary gate failed ({type(exc).__name__}: {exc})")
                results = self._predict(frame)
            else:
                raise

        result = results[0]
        detections: list[dict[str, Any]] = []

        if result.boxes is None or len(result.boxes) == 0:
            return False, detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            bbox = xyxy[i].tolist()
            area = _bbox_area(bbox)

            if area < self.min_box_area:
                continue

            detections.append({
                "bbox": bbox,
                "score": float(confs[i]),
                "class_id": int(cls_ids[i]),
                "area": float(area),
            })

        return len(detections) > 0, detections

    @staticmethod
    def draw_gate_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> None:
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            score = float(det.get("score", 0.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(
                frame,
                f"person gate {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )
