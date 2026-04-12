# API quản lý trạng thái service và nạp cấu hình camera từ resources.txt.
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import database
from app.schemas import (
    ConfigCameraOut,
    ConfigStatusOut,
    ResourcesUploadOut,
    ServiceOut,
)
from app.services.camera_runtime import camera_runtime
from app.services.rtsp_manager import rtsp_manager
from app.utils.resources_loader import mask_rtsp_source

router = APIRouter()


def _service_payload() -> ServiceOut:
    return ServiceOut(
        status=camera_runtime.status,
        cameras_total=len(rtsp_manager.cameras),
        cameras_online=rtsp_manager.cameras_online_count(),
    )


@router.get("/api/service", response_model=ServiceOut)
async def get_service() -> ServiceOut:
    return _service_payload()


@router.post("/api/service/start", response_model=ServiceOut)
async def start_service() -> ServiceOut:
    try:
        await camera_runtime.start()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _service_payload()


@router.post("/api/service/stop", response_model=ServiceOut)
async def stop_service() -> ServiceOut:
    await camera_runtime.stop()
    return _service_payload()


@router.get("/api/config/resources/status", response_model=ConfigStatusOut)
async def get_resources_status() -> ConfigStatusOut:
    stored = await database.fetch_camera_resources()
    return ConfigStatusOut(
        config_loaded=stored is not None,
        file_name=stored.file_name if stored else None,
        uploaded_at=stored.uploaded_at if stored else None,
        cameras_configured=len(rtsp_manager.cameras),
    )


@router.post("/api/config/resources/upload", response_model=ResourcesUploadOut)
async def upload_resources(file: UploadFile = File(...)) -> ResourcesUploadOut:
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt resources files are supported")

    try:
        decoded_bytes = await file.read()
        decoded_text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid resources payload. Expected a UTF-8 text file.",
        )

    try:
        configs, skipped_lines, service_status, uploaded_at = await camera_runtime.replace_resources_text(
            file.filename,
            decoded_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError:
        raise HTTPException(status_code=500, detail="Unable to update private camera configuration")

    return ResourcesUploadOut(
        file_name=file.filename,
        skipped_lines=skipped_lines,
        cameras=[
            ConfigCameraOut(
                id=config.id,
                name=config.name,
                masked_rtsp_source=mask_rtsp_source(config.rtsp_url),
            )
            for config in configs
        ],
        service_status=service_status,
        uploaded_at=uploaded_at,
    )
