"""
PoseManager: dùng RTMPose làm primary, tự fallback sang YOLO11n-pose.
API duy nhất: estimate(frame, bbox) → [17,3] hoặc None
"""
import numpy as np
from src.pose.rtmpose_estimator import RTMPoseEstimator
from src.pose.yolo_pose_estimator import YOLOPoseEstimator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PoseManager:
    """
    Primary  : RTMPose ONNX (CPU)
    Fallback : YOLO11n-Pose (CPU)
    
    Tự động switch khi primary fail ≥3 lần liên tiếp.
    Tự reset về primary sau 300 frame.
    """
    FAIL_THRESHOLD  = 3
    RESET_INTERVAL  = 300

    def __init__(self,
                 rtmpose_onnx: str,
                 yolo_pose_pt: str,
                 device: str = "cpu"):

        self._primary  = RTMPoseEstimator(rtmpose_onnx, device)
        self._fallback = YOLOPoseEstimator(yolo_pose_pt, device)
        self._fail_count     = 0
        self._using_fallback = False
        self._frame_count    = 0

        if not self._primary.is_ready:
            logger.warning(
                "RTMPose không load được → dùng YOLO11n-Pose"
            )
            self._using_fallback = True

        self._log_status()

    def _log_status(self):
        primary_ok  = "✅" if self._primary.is_ready  else "❌"
        fallback_ok = "✅" if self._fallback.is_ready else "❌"
        active = "YOLO11n-Pose" if self._using_fallback else "RTMPose"
        logger.info(
            f"PoseManager: RTMPose={primary_ok} "
            f"YOLO-fallback={fallback_ok} "
            f"Active={active}"
        )

    def estimate(self, frame_bgr: np.ndarray,
                 bbox: list) -> np.ndarray | None:
        self._frame_count += 1

        # Reset về primary định kỳ
        if (self._using_fallback
                and self._primary.is_ready
                and self._frame_count % self.RESET_INTERVAL == 0):
            self._using_fallback = False
            self._fail_count     = 0
            logger.info("PoseManager: thử lại RTMPose primary")

        # Chọn estimator
        estimator = (self._fallback if self._using_fallback
                     else self._primary)

        result = estimator.estimate(frame_bgr, bbox)

        if result is None:
            self._fail_count += 1
            if (not self._using_fallback
                    and self._fail_count >= self.FAIL_THRESHOLD
                    and self._fallback.is_ready):
                self._using_fallback = True
                logger.warning(
                    f"RTMPose fail {self._fail_count}× → "
                    f"switch sang YOLO11n-Pose fallback"
                )
            # Thử fallback ngay frame này
            if self._fallback.is_ready:
                result = self._fallback.estimate(frame_bgr, bbox)
        else:
            self._fail_count = 0

        return result

    def estimate_batch(self, frame_bgr: np.ndarray,
                       bboxes: list) -> list:
        return [self.estimate(frame_bgr, bb) for bb in bboxes]

    @property
    def active_backend(self) -> str:
        return "yolo_fallback" if self._using_fallback else "rtmpose"
