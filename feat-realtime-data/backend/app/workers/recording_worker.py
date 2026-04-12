# Timer hậu ghi dùng để kết thúc clip sau khoảng post-roll của mỗi camera.
"""
RecordingWorker: manages the post-roll timer for a single camera clip.

Runs an asyncio task that:
  1. Waits for CLIP_SECONDS_AFTER after the last detection event.
  2. If another detection arrives before timeout, resets the timer.
  3. On timeout, calls recorder_service.stop_clip() and emits events.

This is kept separate from CameraWorker to allow clean timer management
without blocking the capture loop thread.

NOTE: In the current default architecture, post-roll is handled inline
in camera_worker.py. This module provides an alternative async approach
for finer control.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, Optional, Any

logger = logging.getLogger(__name__)


class PostRollTimer:
    """
    Async post-roll countdown timer.
    Call extend() each time a new detection arrives.
    The on_expire coroutine is called once the timer fires without extension.
    """

    def __init__(
        self,
        delay_secs: float,
        on_expire: Callable[[], Coroutine[Any, Any, None]],
        camera_id: str = "",
    ) -> None:
        self._delay = delay_secs
        self._on_expire = on_expire
        self._camera_id = camera_id
        self._task: Optional[asyncio.Task] = None

    def start_or_extend(self) -> None:
        """Reset the countdown. Call on every new detection event."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._countdown())

    def cancel(self) -> None:
        """Cancel without firing on_expire."""
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    async def _countdown(self) -> None:
        try:
            await asyncio.sleep(self._delay)
            logger.debug("[%s] Post-roll timer expired", self._camera_id)
            await self._on_expire()
        except asyncio.CancelledError:
            pass  # timer was extended or cancelled — normal
