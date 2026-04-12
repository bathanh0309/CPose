# API cho hàng đợi pose/ADL sau khi clip raw đã được lưu.
from fastapi import APIRouter, HTTPException

from app.schemas import ProcessingQueueOut
from app.services.processing_service import processing_service

router = APIRouter()


@router.get("/api/processing/queue", response_model=ProcessingQueueOut)
async def get_processing_queue() -> ProcessingQueueOut:
    return ProcessingQueueOut(**processing_service.snapshot())


@router.post("/api/processing/jobs/{raw_file_name}/retry", status_code=204)
async def retry_processing_job(raw_file_name: str) -> None:
    retried = processing_service.retry(raw_file_name)
    if not retried:
        raise HTTPException(
            status_code=404,
            detail=f"Failed job '{raw_file_name}' not found",
        )
