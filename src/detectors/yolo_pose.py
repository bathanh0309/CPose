from pathlib import Path

from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YoloPoseTracker:
    def __init__(
        self,
        weights,
        conf=0.5,
        iou=0.5,
        tracker="bytetrack.yaml",
        device=None,
        classes=None
    ):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLO pose weights not found: {self.weights}\n"
                "Download yolo11n-pose.pt and set [pose].weights in configs/system/pipeline.yaml."
            )
        self.conf = float(conf)
        self.iou = float(iou)
        self.tracker = tracker
        self.device = device
        self.classes = classes if classes is not None else [0]
        logger.info(f"Loading YOLO pose model: {self.weights}")
        logger.info(f"YOLO device: {self.device}")
        logger.info(f"YOLO tracker: {self.tracker}")
        self.model = YOLO(str(self.weights))

    def infer(self, frame, persist=True):
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections, result

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()

        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        kp_xy = None
        kp_conf = None
        if result.keypoints is not None:
            kp_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                kp_conf = result.keypoints.conf.cpu().numpy()

        for i in range(len(xyxy)):
            det = {
                "bbox": xyxy[i].tolist(),
                "score": float(confs[i]),
                "class_id": int(cls_ids[i]),
                "track_id": int(track_ids[i]) if track_ids is not None else -1,
                "keypoints": kp_xy[i].tolist() if kp_xy is not None else None,
                "keypoint_scores": kp_conf[i].tolist() if kp_conf is not None else None
            }
            detections.append(det)

        return detections, result
