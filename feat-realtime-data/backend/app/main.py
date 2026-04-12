# Điểm vào FastAPI của backend realtime, websocket sự kiện và static dashboard.
"""
FastAPI application entry point.

Startup sequence:
  1. Init SQLite database
  2. Load private camera configs from backend storage
  3. Load detector model
  4. Serve API + WebSocket (camera workers start only after /api/service/start)

All worker threads are daemon threads - they exit when the main process exits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Suppress noisy FFmpeg/OpenCV logs before cv2 is imported anywhere downstream.
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import database
from app.api import cameras, events, health, preview, processing, system
from app.config import get_settings
from app.schemas import WsPingMessage
from app.services.camera_runtime import camera_runtime
from app.services.detector_service import detector_service
from app.services.event_bus import event_bus
from app.services.processing_service import processing_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
_settings = get_settings()
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STATIC_DIR = _REPO_ROOT / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Setup on startup, teardown on shutdown."""
    logger.info("=== RTSP Monitor Backend starting (env=%s) ===", _settings.app_env)

    _settings.recordings_dir.mkdir(parents=True, exist_ok=True)
    _settings.processed_videos_dir.mkdir(parents=True, exist_ok=True)

    await database.init_db()
    logger.info("Database ready: %s", _settings.db_path)

    detector_service.load()
    logger.info("Detector mode: %s", detector_service.mode)
    processing_service.initialize()

    loop = asyncio.get_running_loop()
    camera_runtime.bind_loop(loop)
    await camera_runtime.initialize()

    heartbeat_task = asyncio.create_task(_ws_heartbeat())

    yield

    heartbeat_task.cancel()
    await camera_runtime.stop()
    processing_service.shutdown()
    logger.info("=== RTSP Monitor Backend stopped ===")


app = FastAPI(
    title="RTSP Monitor API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        _settings.frontend_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(preview.router)
app.include_router(processing.router)
app.include_router(system.router)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
app.mount(
    "/processed",
    StaticFiles(directory=_settings.processed_videos_dir, check_dir=False),
    name="processed_videos",
)


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(_INDEX_FILE)


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    """
    Real-time event and camera status stream.
    Clients receive JSON messages with shape:
      { "type": "event" | "camera_status" | "ping", "payload": {...} }
    """
    await event_bus.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await event_bus.disconnect(ws)


async def _ws_heartbeat() -> None:
    """Send a ping every 20s to keep WebSocket connections alive through proxies."""
    ping = WsPingMessage().model_dump_json()
    while True:
        await asyncio.sleep(20)
        await event_bus._broadcast(ping)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=_settings.api_port,
        reload=_settings.is_dev,
        log_level="info",
    )
