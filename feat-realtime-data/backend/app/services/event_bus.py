# Bus phát sự kiện realtime qua WebSocket và đồng bộ xuống database.
"""
Event bus: persists events to SQLite and broadcasts them to all connected
WebSocket clients in real time.

Design:
- Services call `event_bus.publish(EventCreate(...))`
- EventBus persists to DB, then fans out to all WS connections
- Thread-safe via asyncio.Queue; no locks needed when run in single event loop
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Set

from fastapi.websockets import WebSocket

from app import database
from app.schemas import (
    EventCreate,
    EventOut,
    WsCameraStatusMessage,
    WsEventMessage,
    CameraStatusUpdate,
)

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ── WebSocket client registry ──────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("WS client connected. Total: %d", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("WS client disconnected. Total: %d", len(self._clients))

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── Event publishing ───────────────────────────────────────────────────

    async def publish(self, evt: EventCreate) -> EventOut:
        """
        Persist event to DB and broadcast to all WS clients.
        Returns the stored EventOut (with generated id).
        """
        stored = await database.insert_event(evt)
        msg = WsEventMessage(payload=stored)
        await self._broadcast(msg.model_dump_json())
        return stored

    async def publish_camera_status(self, update: CameraStatusUpdate) -> None:
        """Broadcast a camera status update without persisting to DB."""
        msg = WsCameraStatusMessage(payload=update)
        await self._broadcast(msg.model_dump_json())

    # ── Internal ──────────────────────────────────────────────────────────

    async def _broadcast(self, text: str) -> None:
        if not self._clients:
            return

        dead: list[WebSocket] = []
        async with self._lock:
            snapshot = list(self._clients)

        for ws in snapshot:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


# Singleton — imported by API routes and services
event_bus = EventBus()
