# Registry giữ cấu hình RTSP và trạng thái sống của toàn bộ camera.
"""
RTSP Manager: central registry of all cameras.
Loads configs from private backend storage, tracks live status, and provides
camera state to API routes.

Each camera's actual streaming is handled by a CameraWorker (workers/camera_worker.py).
This module only manages metadata and status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from app import database
from app.config import get_settings
from app.schemas import CameraOut, StreamFormat
from app.utils.resources_loader import CameraConfig, mask_rtsp_source, parse_cameras_text

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class CameraState:
    config: CameraConfig
    online: bool = False
    source_fps: float = 0.0
    detect_fps: float = 0.0
    preview_fps: float = 0.0
    decoder_errors: int = 0
    preview_enabled: bool = False
    last_seen_at: Optional[datetime] = None

    def mark_online(self) -> None:
        if not self.online:
            self.online = True
        self.last_seen_at = datetime.utcnow()

    def mark_offline(self) -> None:
        self.online = False
        self.source_fps = 0.0
        self.detect_fps = 0.0
        self.preview_fps = 0.0

    def to_api(self) -> CameraOut:
        return CameraOut(
            id=self.config.id,
            name=self.config.name,
            masked_rtsp_source=mask_rtsp_source(self.config.rtsp_url),
            rtsp_configured=bool(self.config.rtsp_url),
            online=self.online,
            source_fps=self.source_fps,
            detect_fps=self.detect_fps,
            preview_fps=self.preview_fps,
            decoder_errors=self.decoder_errors,
            preview_enabled=self.preview_enabled,
            preview_url=f"/api/cameras/{self.config.id}/preview",
            stream_format=StreamFormat.MJPEG,
            last_seen_at=self.last_seen_at.isoformat() + "Z"
            if self.last_seen_at
            else None,
        )


class RtspManager:
    """
    Singleton that holds camera configs and their live state.
    Workers call update methods here after each captured frame.
    """

    def __init__(self) -> None:
        self._cameras: Dict[str, CameraState] = {}

    async def load(self) -> None:
        """Load camera list from the private backend configuration store."""
        stored = await database.fetch_camera_resources()
        if stored is None:
            self.replace_configs([])
            logger.info("No camera configuration has been uploaded yet.")
            return

        configs, skipped_lines = parse_cameras_text(
            stored.content,
            max_cameras=_settings.max_cameras,
        )
        self.replace_configs(configs)

        if not configs:
            logger.warning(
                "Stored camera configuration '%s' contains no valid cameras.",
                stored.file_name,
            )
        else:
            logger.info(
                "Loaded %d camera(s) from private config '%s' (skipped=%d)",
                len(configs),
                stored.file_name,
                skipped_lines,
            )

    def replace_configs(self, configs: list[CameraConfig]) -> None:
        self._cameras = {cfg.id: CameraState(config=cfg) for cfg in configs}

    @property
    def cameras(self) -> list[CameraState]:
        return list(self._cameras.values())

    def get(self, camera_id: str) -> Optional[CameraState]:
        return self._cameras.get(camera_id)

    def all_api(self) -> list[CameraOut]:
        return [cam.to_api() for cam in self._cameras.values()]

    def update_metrics(
        self,
        camera_id: str,
        source_fps: float,
        detect_fps: float,
        preview_fps: float,
        decoder_errors: int,
    ) -> None:
        cam = self._cameras.get(camera_id)
        if cam:
            cam.source_fps = source_fps
            cam.detect_fps = detect_fps
            cam.preview_fps = preview_fps
            cam.decoder_errors = decoder_errors
            cam.mark_online()

    def mark_online(
        self,
        camera_id: str,
        source_fps: Optional[float] = None,
        detect_fps: Optional[float] = None,
        preview_fps: Optional[float] = None,
    ) -> None:
        cam = self._cameras.get(camera_id)
        if cam:
            if source_fps is not None:
                cam.source_fps = source_fps
            if detect_fps is not None:
                cam.detect_fps = detect_fps
            if preview_fps is not None:
                cam.preview_fps = preview_fps
            cam.mark_online()

    def on_offline(self, camera_id: str) -> None:
        cam = self._cameras.get(camera_id)
        if cam:
            cam.mark_offline()

    def set_preview_enabled(self, camera_id: str, enabled: bool) -> None:
        cam = self._cameras.get(camera_id)
        if cam:
            cam.preview_enabled = enabled

    def select_preview_camera(self, camera_id: Optional[str]) -> None:
        for cam in self._cameras.values():
            cam.preview_enabled = camera_id is not None and cam.config.id == camera_id

    def cameras_online_count(self) -> int:
        return sum(1 for c in self._cameras.values() if c.online)


rtsp_manager = RtspManager()
