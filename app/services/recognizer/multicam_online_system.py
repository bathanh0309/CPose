from __future__ import annotations

import time
from typing import Optional

from app.core.tracking_types import Track


class SingleCamProcessor:
    """Placeholder processor for one live camera stream in the Flask app pipeline."""

    def __init__(self, cam_config: dict):
        self.cam_id = cam_config["id"]
        self._last_timestamp: float = 0.0
        self._current_tracks: list[Track] = []

    def process_frame(self, frame, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        self._last_timestamp = timestamp
        self._current_tracks = []

    def get_current_tracks(self) -> tuple[float, list[Track]]:
        return self._last_timestamp, list(self._current_tracks)


class MultiCamOnlineSystem:
    """App-side live multi-camera orchestrator."""

    def __init__(self, config):
        self.config = config
        self._cam_processors: dict[str, SingleCamProcessor] = {}
        self._last_global_timestamp: float = 0.0
        self._build_from_config(config)

    def _build_from_config(self, config) -> None:
        cameras_cfg = config.get("cameras", []) if isinstance(config, dict) else []
        if not cameras_cfg and isinstance(config, dict) and "camera_streams" in config:
            cameras_cfg = [
                {"id": f"cam{index:02d}", "url": url}
                for index, url in enumerate(config["camera_streams"])
            ]

        for cam_cfg in cameras_cfg:
            cam_id = cam_cfg["id"]
            self._cam_processors[cam_id] = SingleCamProcessor(cam_cfg)

    def process_frame(self, cam_id: str, frame, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        processor = self._cam_processors.get(cam_id)
        if processor:
            processor.process_frame(frame, timestamp)
            self._last_global_timestamp = timestamp

    def get_current_tracks(self) -> tuple[float, list[Track]]:
        all_tracks: list[Track] = []
        latest_ts = 0.0
        for processor in self._cam_processors.values():
            cam_ts, cam_tracks = processor.get_current_tracks()
            latest_ts = max(latest_ts, cam_ts)
            all_tracks.extend(cam_tracks)
        if latest_ts == 0.0:
            latest_ts = self._last_global_timestamp
        return latest_ts, all_tracks

