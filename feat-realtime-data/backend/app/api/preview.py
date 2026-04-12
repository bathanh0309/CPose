# API stream MJPEG để xem trước hình ảnh trực tiếp của từng camera.
"""
MJPEG Preview endpoint.

Browser connects to /api/cameras/{id}/preview and receives a
multipart/x-mixed-replace stream of JPEG frames.

Architecture note:
- This endpoint is the ONLY place where preview frames leave the backend.
- It reads from preview_streamer (a frame cache populated by CameraWorker).
- Stopping/starting preview from the UI does NOT affect detection or recording.
- If no cameras are online, a blank frame is served — never an error.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.preview_streamer import preview_streamer
from app.services.rtsp_manager import rtsp_manager

router = APIRouter()

_BOUNDARY = "frame"
_CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={_BOUNDARY}"


@router.get("/api/cameras/{camera_id}/preview")
async def mjpeg_preview(camera_id: str) -> StreamingResponse:
    # 404 if camera is not configured at all
    if rtsp_manager.get(camera_id) is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    return StreamingResponse(
        preview_streamer.frame_generator(camera_id),
        media_type=_CONTENT_TYPE,
        headers={
            # Prevent any proxy / browser from caching the live stream
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            # Keep connection alive for stream consumers
            "Connection": "keep-alive",
        },
    )
