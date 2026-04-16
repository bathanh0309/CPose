from typing import Dict, List, Tuple, Optional
import time

from cpose.core.tracking.types import Track

class SingleCamProcessor:
    """
    Handles processing for a single camera:
    - Person Detection
    - Tracking
    - Pose Estimation
    - ADL Inference
    - ReID & FaceID
    """

    def __init__(self, cam_config):
        self.cam_id = cam_config["id"]
        self._last_timestamp: float = 0.0
        
        # Placeholders for research/app modules
        # self.detector = ...
        # self.tracker = ...
        # self.pose_model = ...
        # self.adl_engine = ...
        
        self._current_tracks: List[Track] = []

    def process_frame(self, frame, timestamp: Optional[float] = None):
        """Update processor with a new frame."""
        if timestamp is None:
            timestamp = time.time()
        self._last_timestamp = timestamp

        # Example Pipeline:
        # 1. detections = self.detector.detect(frame)
        # 2. tracks = self.tracker.update(detections, timestamp)
        # 3. for trk in tracks: self._update_pose/adl/face(trk, frame)
        
        # For now, placeholder for active tracks
        self._current_tracks = []

    def get_current_tracks(self) -> Tuple[float, List[Track]]:
        """Return latest timestamp and active tracks for this camera."""
        return self._last_timestamp, list(self._current_tracks)


class MultiCamOnlineSystem:
    """
    Systems-level orchestrator for multiple cameras.
    Maintains the state of all active camera processors.
    """

    def __init__(self, config):
        self.config = config
        self._cam_processors: Dict[str, SingleCamProcessor] = {}
        self._last_global_timestamp: float = 0.0
        self._build_from_config(config)

    def _build_from_config(self, config):
        """Initialize processors for each camera defined in config."""
        # Config usually contains a list of camera definitions
        cameras_cfg = config.get("cameras", [])
        if not cameras_cfg and "camera_streams" in config: # Handle common aliases
            cameras_cfg = [{"id": f"cam{i:02d}", "url": url} for i, url in enumerate(config["camera_streams"])]

        for cam_cfg in cameras_cfg:
            cam_id = cam_cfg["id"]
            self._cam_processors[cam_id] = SingleCamProcessor(cam_cfg)

    def process_frame(self, cam_id: str, frame, timestamp: Optional[float] = None):
        """Push a frame into the specific camera processor."""
        if timestamp is None:
            timestamp = time.time()

        processor = self._cam_processors.get(cam_id)
        if processor:
            processor.process_frame(frame, timestamp)
            self._last_global_timestamp = timestamp

    def get_current_tracks(self) -> Tuple[float, List[Track]]:
        """Snapshot of all active tracks across the entire system."""
        all_tracks: List[Track] = []
        latest_ts = 0.0

        for processor in self._cam_processors.values():
            cam_ts, cam_tracks = processor.get_current_tracks()
            latest_ts = max(latest_ts, cam_ts)
            all_tracks.extend(cam_tracks)

        if latest_ts == 0.0:
            latest_ts = self._last_global_timestamp

        return latest_ts, all_tracks
