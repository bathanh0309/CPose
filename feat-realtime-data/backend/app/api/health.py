# API healthcheck trả uptime, số camera online và số client WebSocket.
from fastapi import APIRouter
import time

from app.schemas import HealthOut
from app.services.event_bus import event_bus
from app.services.rtsp_manager import rtsp_manager

router = APIRouter()

_START_TIME = time.monotonic()


@router.get("/api/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(
        status="ok",
        uptime=round(time.monotonic() - _START_TIME, 1),
        cameras_online=rtsp_manager.cameras_online_count(),
        cameras_total=len(rtsp_manager.cameras),
        ws_clients=event_bus.client_count,
    )
