"""YOLO Detect + Pose wrappers using Ultralytics (PyTorch .pt models).

Drop-in replacement for the former yolo_openvino.py module.
Uses ultralytics YOLO for both detection and pose estimation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class YOLODetectUltralytics:
    """Person detection using Ultralytics YOLO .pt model."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        class_filter: list[int] | None = None,
        imgsz: int = 640,
    ):
        self.device = str(device)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_filter = list(class_filter or [0])
        self.imgsz = int(imgsz)
        self.last_latency_ms = 0.0

        self.model = YOLO(str(weights))
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """Run detection. Returns list of [x1, y1, x2, y2, score]."""
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.class_filter,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        bboxes: list[list[float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                bboxes.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), score])
        return bboxes


class YOLOPoseUltralytics:
    """Pose estimation using Ultralytics YOLO-Pose .pt model."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ):
        self.device = str(device)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.last_latency_ms = 0.0

        self.model = YOLO(str(weights))
        self.model.to(self.device)

    def estimate(self, frame: np.ndarray, bboxes: list[Any]) -> list[dict[str, Any]]:
        """Run pose on cropped regions from pre-detected bboxes.

        Parameters
        ----------
        frame : Full frame image.
        bboxes : List of detections — either dicts with 'bbox'/'score'/'track_id'
                 or raw lists [x1, y1, x2, y2, score].

        Returns
        -------
        List of detection dicts with keypoints.
        """
        persons: list[dict[str, Any]] = []
        h, w = frame.shape[:2]

        for item in bboxes:
            bbox, score, track_id = self._read_bbox_item(item)
            x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            t0 = time.perf_counter()
            results = self.model.predict(
                crop,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

            keypoints = np.zeros((17, 3), dtype=np.float32)
            for r in results:
                if r.keypoints is not None and len(r.keypoints) > 0:
                    kps = r.keypoints[0]
                    xy = kps.xy[0].cpu().numpy()  # (17, 2)
                    conf_arr = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(17, dtype=np.float32)
                    if xy.shape[0] == 17:
                        keypoints[:, 0] = xy[:, 0] + float(x1)
                        keypoints[:, 1] = xy[:, 1] + float(y1)
                        keypoints[:, 2] = conf_arr
                    break

            persons.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
                "class_id": 0,
                "keypoints": keypoints,
                "keypoint_scores": keypoints[:, 2].tolist(),
            })
        return persons

    @staticmethod
    def _read_bbox_item(item: Any):
        if isinstance(item, dict):
            bbox = item.get("bbox", [0, 0, 0, 0])
            return bbox, float(item.get("score", item.get("conf", 0.0))), int(item.get("track_id", -1))
        return item[:4], float(item[4] if len(item) > 4 else 0.0), -1
