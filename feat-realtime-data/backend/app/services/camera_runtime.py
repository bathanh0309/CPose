# Điều phối vòng đời worker camera khi start, stop hoặc đổi cấu hình.
from __future__ import annotations

import asyncio
import logging
from typing import Dict

from app import database
from app.config import get_settings
from app.schemas import ServiceState
from app.services.preview_streamer import preview_streamer
from app.services.recorder_service import recorder_service
from app.services.rtsp_manager import rtsp_manager
from app.utils.resources_loader import CameraConfig, parse_cameras_text
from app.workers.camera_worker import CameraWorker

logger = logging.getLogger(__name__)
_settings = get_settings()


class CameraRuntime:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: Dict[str, CameraWorker] = {}
        self._running = False
        self._lock = asyncio.Lock()

    @property
    def status(self) -> ServiceState:
        return ServiceState.active if self._running else ServiceState.stopped

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def initialize(self) -> None:
        async with self._lock:
            await rtsp_manager.load()
            preview_streamer.reset()
            self._running = False

    async def start(self) -> ServiceState:
        async with self._lock:
            if not rtsp_manager.cameras:
                raise ValueError("No valid cameras found. Upload resources.txt first.")
            await asyncio.to_thread(self._start_workers_sync)
            return self.status

    async def stop(self) -> ServiceState:
        async with self._lock:
            await asyncio.to_thread(self._stop_workers_sync)
            return self.status

    async def replace_resources_text(
        self,
        file_name: str,
        text: str,
    ) -> tuple[list[CameraConfig], int, ServiceState, str]:
        configs, skipped_lines = parse_cameras_text(
            text,
            max_cameras=_settings.max_cameras,
        )
        if not configs:
            raise ValueError("No valid cameras found in resources.txt")

        normalized_text = text.replace("\r\n", "\n").strip()
        if normalized_text:
            normalized_text = f"{normalized_text}\n"

        async with self._lock:
            was_running = self._running
            if was_running:
                await asyncio.to_thread(self._stop_workers_sync)

            stored = await database.save_camera_resources(file_name, normalized_text)
            rtsp_manager.replace_configs(configs)
            preview_streamer.reset()

            if was_running:
                await asyncio.to_thread(self._start_workers_sync)

            return configs, skipped_lines, self.status, stored.uploaded_at

    def _start_workers_sync(self) -> None:
        if self._running:
            return
        if self._loop is None:
            raise RuntimeError("Camera runtime loop is not bound")

        self._workers = {}
        for cam_state in rtsp_manager.cameras:
            worker = CameraWorker(config=cam_state.config, loop=self._loop)
            worker.start()
            self._workers[cam_state.config.id] = worker

        self._running = True
        logger.info("Camera service active with %d worker(s)", len(self._workers))

    def _stop_workers_sync(self) -> None:
        if self._workers:
            for worker in self._workers.values():
                worker.stop()

        recorder_service.stop_all()
        preview_streamer.reset()

        for cam_state in rtsp_manager.cameras:
            cam_state.preview_enabled = False
            cam_state.mark_offline()

        self._workers = {}
        self._running = False
        logger.info("Camera service stopped")


camera_runtime = CameraRuntime()
