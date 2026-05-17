"""
RTMPose top-down estimator.
Input : frame BGR + bbox [x1,y1,x2,y2]
Output: keypoints [17,3] (x,y,conf) — COCO 17 format
"""
import numpy as np
import cv2
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# COCO 17 keypoint order (giống YOLO11n-pose — không đổi gì downstream)
COCO17_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

class RTMPoseEstimator:
    """
    Wrapper RTMPose ONNX — top-down, input 256×192.
    Chạy hoàn toàn CPU (onnxruntime).
    """

    INPUT_W  = 192
    INPUT_H  = 256
    NUM_KPS  = 17

    def __init__(self, onnx_path: str, device: str = "cpu"):
        self.onnx_path = Path(onnx_path)
        self._session  = None
        self._ready    = False
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.inter_op_num_threads = 2
            so.intra_op_num_threads = 4
            so.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(self.onnx_path), so,
                providers=providers
            )
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._ready = True
            logger.info(f"RTMPose loaded: {self.onnx_path.name}")
        except Exception as e:
            logger.error(f"RTMPose load FAILED: {e}")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Preprocessing ───────────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray,
                    bbox: list) -> tuple[np.ndarray | None, dict]:
        """
        Crop + pad + resize về 256×192.
        Trả về (input_tensor [1,3,256,192], meta) để unproject sau.
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Clamp bbox vào frame
        H, W = frame_bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, {}

        ch, cw = crop.shape[:2]
        # Pad thành tỉ lệ 3:4 (W:H = 192:256)
        target_ratio = self.INPUT_W / self.INPUT_H   # 0.75
        curr_ratio   = cw / (ch + 1e-6)
        if curr_ratio > target_ratio:
            new_h = int(cw / target_ratio)
            pad   = (new_h - ch) // 2
            crop  = cv2.copyMakeBorder(
                crop, pad, pad, 0, 0,
                cv2.BORDER_CONSTANT, value=(128,128,128)
            )
            y1 -= pad
        else:
            new_w = int(ch * target_ratio)
            pad   = (new_w - cw) // 2
            crop  = cv2.copyMakeBorder(
                crop, 0, 0, pad, pad,
                cv2.BORDER_CONSTANT, value=(128,128,128)
            )
            x1 -= pad

        resized = cv2.resize(crop, (self.INPUT_W, self.INPUT_H))
        # BGR → RGB, normalize
        img = resized[:,:,::-1].astype(np.float32)
        img = (img - np.array([123.675, 116.28, 103.53])) / \
               np.array([58.395,  57.12,  57.375])
        tensor = img.transpose(2,0,1)[None].astype(np.float32)

        meta = {
            "x1": x1, "y1": y1,
            "crop_w": crop.shape[1],
            "crop_h": crop.shape[0],
        }
        return tensor, meta

    # ── Postprocessing ──────────────────────────────────────────
    def _postprocess(self, output: np.ndarray,
                     meta: dict) -> np.ndarray:
        """
        output shape: [1,17,2] hoặc [1,17,3] tùy model SimCC
        Trả về [17,3] (x,y,conf) trong tọa độ frame gốc.
        """
        out = output[0]          # [17,2] hoặc [17,3]
        if out.shape[-1] == 2:
            # SimCC: (x,y) ở scale INPUT
            kps_x = out[:,0] / self.INPUT_W  * meta["crop_w"] + meta["x1"]
            kps_y = out[:,1] / self.INPUT_H  * meta["crop_h"] + meta["y1"]
            conf  = np.ones(self.NUM_KPS, dtype=np.float32)
        else:
            kps_x = out[:,0] / self.INPUT_W  * meta["crop_w"] + meta["x1"]
            kps_y = out[:,1] / self.INPUT_H  * meta["crop_h"] + meta["y1"]
            conf  = out[:,2]

        result = np.stack([kps_x, kps_y, conf], axis=1).astype(np.float32)
        return result   # [17,3]

    # ── Public API ──────────────────────────────────────────────
    def estimate(self, frame_bgr: np.ndarray,
                 bbox: list) -> np.ndarray | None:
        """
        Trả về [17,3] keypoints hoặc None nếu lỗi.
        """
        if not self._ready:
            return None
        tensor, meta = self._preprocess(frame_bgr, bbox)
        if tensor is None:
            return None
        try:
            output = self._session.run(
                [self._output_name],
                {self._input_name: tensor}
            )
            return self._postprocess(output[0], meta)
        except Exception as e:
            logger.warning(f"RTMPose infer error: {e}")
            return None

    def estimate_batch(self, frame_bgr: np.ndarray,
                       bboxes: list[list]) -> list[np.ndarray | None]:
        """Xử lý nhiều người cùng lúc."""
        return [self.estimate(frame_bgr, bb) for bb in bboxes]
