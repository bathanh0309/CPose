"""
YOLO11n-Pose estimator — dùng làm fallback khi RTMPose fail.
Output chuẩn hóa giống RTMPoseEstimator: [17,3] (x,y,conf).
"""
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class YOLOPoseEstimator:
    def __init__(self, model_path: str, device: str = "cpu"):
        self._model  = None
        self._ready  = False
        self._device = device
        self._path   = model_path
        self._load()

    def _load(self):
        try:
            self._model = YOLO(self._path)
            self._ready = True
            logger.info(f"YOLOPose fallback loaded: {Path(self._path).name}")
        except Exception as e:
            logger.error(f"YOLOPose load FAILED: {e}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def estimate(self, frame_bgr, bbox: list) -> np.ndarray | None:
        if not self._ready:
            return None
        try:
            results = self._model.predict(
                frame_bgr, imgsz=640,
                conf=0.55, device=self._device,
                verbose=False, classes=[0]
            )
            if not results or results[0].keypoints is None:
                return None
            kps = results[0].keypoints.data  # [N,17,3]
            if kps.shape[0] == 0:
                return None
            # Lấy person gần bbox nhất
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            cx, cy  = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            dists   = ((boxes[:,0]+boxes[:,2])/2 - cx)**2 + \
                      ((boxes[:,1]+boxes[:,3])/2 - cy)**2
            best    = int(dists.argmin())
            return kps[best].cpu().numpy().astype(np.float32)  # [17,3]
        except Exception as e:
            logger.warning(f"YOLOPose fallback error: {e}")
            return None
