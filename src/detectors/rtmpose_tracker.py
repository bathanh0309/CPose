from pathlib import Path
import numpy as np

from src.detectors.pedestrian_yolo import PedestrianYoloTracker
from src.utils.filters import DetectionFilterStats, filter_person_detections
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RTMPoseTopDownTracker:
    """Top-down pose tracker using a detector (YOLO pedestrian) + RTMPose for keypoints.

    This class is a drop-in alternative to `YoloPoseTracker` but reuses the
    existing pedestrian detector/tracker for stable track ids, then runs
    RTMPose per bbox to obtain keypoints.
    """

    def __init__(
        self,
        pose_model: str,
        det_weights: str,
        conf: float = 0.4,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
        device: str = "cpu",
        backend: str = "onnxruntime",
        tracking_cfg: dict | None = None,
        classes: list | None = None,
    ):
        self.pose_model_path = str(pose_model) if pose_model is not None else None
        self.tracking_cfg = tracking_cfg or {}
        # Use pedestrian YOLO (existing) for detection+tracking
        self.detector = PedestrianYoloTracker(
            weights=det_weights,
            conf=conf,
            iou=iou,
            tracker=tracker,
            device=device,
            classes=classes,
            tracking_cfg=tracking_cfg,
        )

        # Load RTMPose (rtmlib) lazily
        try:
            from rtmlib import RTMPose

            if not self.pose_model_path:
                raise FileNotFoundError("RTMPose model path not provided in config")
            logger.info(f"Loading RTMPose model: {self.pose_model_path} (backend={backend})")
            self.pose = RTMPose(pose=self.pose_model_path, device=device, backend=backend)
        except Exception as exc:
            raise ImportError(
                "RTMPose (rtmlib) initialization failed. Install rtmlib or provide a valid model. "
                "Install via `pip install rtmlib` and set pose.rtmpose.model in config."
            ) from exc

        self.last_filter_stats = DetectionFilterStats()

    def infer(self, frame, persist=True):
        """Run detector+tracker, then RTMPose per bbox.

        Returns: (detections, raw_result) matching other detectors' contract.
        """
        detections, result = self.detector.infer(frame, persist=persist)

        if not detections:
            return detections, result

        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                det["keypoints"] = None
                det["keypoint_scores"] = None
                continue
            try:
                # RTMPose expects (frame, bbox)
                kps = self.pose.infer(frame, bbox)
                if kps is None:
                    det["keypoints"] = None
                    det["keypoint_scores"] = None
                    continue
                arr = np.asarray(kps)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    keypoints = arr[:, :2].astype(float).tolist()
                    det["keypoints"] = keypoints
                    if arr.shape[1] >= 3:
                        det["keypoint_scores"] = arr[:, 2].astype(float).tolist()
                    else:
                        det["keypoint_scores"] = [1.0] * len(keypoints)
                else:
                    det["keypoints"] = None
                    det["keypoint_scores"] = None
            except Exception as exc:
                logger.warning(f"RTMPose inference failed for bbox={bbox}: {exc}")
                det["keypoints"] = None
                det["keypoint_scores"] = None

        detections, self.last_filter_stats = filter_person_detections(detections, self.tracking_cfg)
        return detections, result
