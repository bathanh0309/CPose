# API liệt kê camera và bật hoặc tắt trạng thái preview trên UI.
from fastapi import APIRouter, HTTPException

from app.schemas import CameraOut
from app.services.rtsp_manager import rtsp_manager

router = APIRouter()


@router.get("/api/cameras", response_model=list[CameraOut])
async def list_cameras() -> list[CameraOut]:
    return rtsp_manager.all_api()


@router.post("/api/cameras/{camera_id}/preview/start", status_code=204)
async def start_preview(camera_id: str) -> None:
    """
    Mark preview as enabled for this camera (UI hint only).
    Detection continues regardless of this call.
    """
    cam = rtsp_manager.get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    rtsp_manager.select_preview_camera(camera_id)


@router.post("/api/cameras/{camera_id}/preview/stop", status_code=204)
async def stop_preview(camera_id: str) -> None:
    """
    Mark preview as disabled for this camera (UI hint only).
    Detection continues regardless of this call.
    """
    cam = rtsp_manager.get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    rtsp_manager.select_preview_camera(None)
