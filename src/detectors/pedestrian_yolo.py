from pathlib import Path

from ultralytics import YOLO

from src.utils.filters import DetectionFilterStats, filter_person_detections
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PedestrianYoloTracker:
    """YOLO pedestrian detector/tracker configured from pipeline.yaml."""

    def __init__(
        self,
        weights,
        conf=0.3,
        iou=0.5,
        tracker="bytetrack.yaml",
        device=None,
        classes=None,
        tracking_cfg=None,
    ):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Pedestrian tracking weights not found: {self.weights}\n"
                "Place the custom YOLO weight at models/tracking.pt or update [pedestrian].weights."
            )

        self.conf = float(conf)
        self.iou = float(iou)
        self.tracker = tracker
        self.device = device
        self.classes = classes
        self.tracking_cfg = tracking_cfg or {}
        self.last_filter_stats = DetectionFilterStats()

        logger.info(f"Loading pedestrian YOLO model: {self.weights}")
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
            verbose=False,
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

        for i in range(len(xyxy)):
            detections.append(
                {
                    "bbox": xyxy[i].tolist(),
                    "score": float(confs[i]),
                    "class_id": int(cls_ids[i]),
                    "track_id": int(track_ids[i]) if track_ids is not None else -1,
                    "keypoints": None,
                    "keypoint_scores": None,
                }
            )

        detections, self.last_filter_stats = filter_person_detections(detections, self.tracking_cfg)
        return detections, result
