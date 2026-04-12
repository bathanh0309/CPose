# API điều khiển Phase 1 (Start/Stop) và Phase 2
import os
import json
import logging
import asyncio

from typing import Optional
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from app.services.phase2_label import (
    extract_human_detection_async,
    get_video_metadata,
    request_stop,
)

from app.utils.file_handler import (
    delete_output_folder,
    delete_temp_file,
    get_output_json_path,
    list_output_subfolders,
    save_upload_temp,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase2", tags=["Phase 2 - Auto Labeling"])

# Job state
# Dict store each job's status in the memory
_jobs: dict[str, dict] = {}

# Pydantic schemas ?? what is this

class ProcessRequest(BaseModel):
    # Body json for endpoint/ process
    videos: list[str] # danh sach temp_path tra ve tu upload
    

class DeleteOutputRequest(BaseModel):
    # Body json for endpoint/ delete-output
    folder_name: str
    
    
# API 1: upload mp4 + extract meta data
@router.post("/upload", summary="Upload file MP4 and return metadata")
def upload_video(file: UploadFile = File(...)):
    """
    Nhận file MP4, lưu vào thư mục temp, đọc metadata cơ bản.

    **Frontend gọi:**
    ```js
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/phase2/upload", { method: "POST", body: fd });
    const { temp_path, fps, duration_sec, ... } = await res.json();
    ```

    **Response:**
    ```json
    { "temp_path": "/data/temp/abc123_cam1.mp4", "fps": 25, "duration_sec": 120.5, ... }
    ```
    """
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only accept .mp4 file")

    try:
        temp_path = save_upload_temp(file.file, file.filename)
    except Exception as e:
        logger.error(f"Save file unsuccessfully: {e}")
        raise HTTPException(status_code=500, detail=f"Save file fail: {e}")
    
    try:
        meta = get_video_metadata(temp_path)
    except ValueError as e:
        delete_temp_file(temp_path)
        raise HTTPException(status_code=422, detail=str(e))
    
    return JSONResponse(content={"temp_path": temp_path, **meta})


# API 2: Start batch processing
MAX_CONCURRENT_JOBS = asyncio.Semaphore(2)

@router.post("/process", summary="Start processing video batch... ")
async def start_processing(body: ProcessRequest, background_tasks: BackgroundTasks):
    
    if not body.videos:
        raise HTTPException(status_code=400, detail="Videos' list empty")
    
    job_ids = []
    
    for temp_path in body.videos:
        original_stem = Path(temp_path).name.split('_',1)[-1].replace('.mp4','')
        job_id = original_stem
        
        job_ids.append(job_id)
        
        if not Path(temp_path).exists():
            logger.warning(f"File doesn't exist: {temp_path}")
            _jobs[job_id] = {"status": "error", "progress": 0, "result": None, "error": "Original file doesn't exist"}
            continue
        
        check_dir = OUTPUT_DIR / original_stem
        if check_dir.exists():
            logger.info(f"Video {original_stem} exists, and had been processed!")
            _jobs[job_id] = {"status":"done", "progress":100, "result": "Processed", "error":None}
            delete_temp_file(temp_path)
            continue
        
        _jobs[job_id] = {"status":"waiting", "progress":0, "result":None, "error":None}
        
        # move video into background processing task
        background_tasks.add_task(
            _run_job,
            job_id=job_id,
            temp_path = temp_path,
            output_folder=str(OUTPUT_DIR),
        )
        
    if not job_ids:
        raise HTTPException(status_code=422, detail="There are no valid file to process")
    
    logger.info(f"Initiated {len(job_ids)} jobs: {job_ids}")
    return JSONResponse(content={"job_ids": job_ids}, status_code=202)

        
async def _run_job(job_id: str, temp_path: str, output_folder: str):
    """
    Hàm worker bắt Background process. Giới hạn xử lý song song nhờ Semaphore.
    """
    # ── Wait/Block ở đây cho tới khi có slot trống ──
    async with MAX_CONCURRENT_JOBS:
        
        # Vượt qua cửa Semaphore nghĩa là bắt đầu tới lượt, cập nhật status!
        if job_id in _jobs:
            _jobs[job_id]["status"] = "processing"
            
        def on_progress(pct: int):
            if job_id in _jobs:
                _jobs[job_id]["progress"] = pct
                
        try:
            result = await extract_human_detection_async(
                video_path=temp_path,
                output_folder=output_folder,
                job_id =job_id,
                progress_cb = on_progress
            )
            
            # Lưu kết quả khi thành công
            if job_id in _jobs:
                _jobs[job_id]["status"]   = "stopped" if result.get("stopped_early") else "done"
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["result"]   = result
                
        except Exception as e:
            logger.error(f"[{job_id}] Lỗi xử lý: {e}", exc_info=True)
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"]  = str(e)
        finally:
            # Xóa file tạm tại /data/temp sau khi hoàn thành (thành công hoặc lỗi)
            delete_temp_file(temp_path)
            
            
# ─── API 3: Polling trạng thái job ───────────────────────────────────────────
@router.get("/status/{job_id}", summary="Kiểm tra trạng thái job (dùng để polling)")
async def get_status(job_id: str):
    """
    UI gọi mỗi 2-3 giây để cập nhật progress bar và trạng thái từng dòng.

    **Response khi đang chạy:**
    ```json
    { "status": "processing", "progress": 45 }
    ```
    **Response khi hoàn tất:**
    ```json
    { "status": "done", "progress": 100, "result": { "total_detections": 32, ... } }
    ```
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' không tồn tại")
    return JSONResponse(content=job)


# ─── API 4: Dừng job đang chạy ───────────────────────────────────────────────
@router.post("/stop/{job_id}", summary="Yêu cầu dừng một job đang chạy")
async def stop_job(job_id: str):
    """
    Gửi stop signal vào job. Worker kiểm tra cờ này mỗi frame và dừng lại.
    Kết quả tạm thời đến thời điểm dừng vẫn được lưu vào JSON.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' không tồn tại")

    if _jobs[job_id]["status"] != "processing":
        raise HTTPException(status_code=400, detail="Job không đang ở trạng thái processing")

    request_stop(job_id)
    logger.info(f"Đã gửi stop signal cho job: {job_id}")
    return {"message": f"Stop signal đã gửi cho job '{job_id}'"}


@router.post("/stop-all", summary="Dừng tất cả jobs đang chạy (nút Stop All trên UI)")
async def stop_all_jobs():
    """Dừng toàn bộ job có status = 'processing'."""
    stopped = []
    for job_id, job in _jobs.items():
        if job["status"] == "processing":
            request_stop(job_id)
            stopped.append(job_id)
    return {"stopped_jobs": stopped, "count": len(stopped)}


# ─── API 5: Lấy kết quả detections ───────────────────────────────────────────
@router.get("/result/{video_name}", summary="Lấy nội dung detections.json của một video")
async def get_result(video_name: str):
    """
    **Ví dụ gọi:**
    ```
    GET /api/phase2/result/cam1_20240101?output_folder=/data/output
    ```
    Trả về toàn bộ nội dung file detections.json.
    """
    json_path = get_output_json_path(str(OUTPUT_DIR), video_name)
    if json_path is None:
        raise HTTPException(status_code=404, detail="Chưa có kết quả cho video này")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Đọc file kết quả thất bại: {e}")


# ─── API 6: Xóa output folder ────────────────────────────────────────────────
@router.delete("/delete-output", summary="Xóa vật lý thư mục output của một video")
async def delete_output(body: DeleteOutputRequest):
    """
    Được gọi khi người dùng bấm nút 🗑️ Delete Output trên UI.
    Xóa folder và reset trạng thái job tương ứng trong memory.
    """
    target = str(OUTPUT_DIR / body.folder_name)
    success = delete_output_folder(target)

    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy folder: {body.folder_name}")

    # Reset job state nếu có (cho phép re-process)
    _jobs.pop(body.folder_name, None)
    logger.info(f"Đã xóa output folder: {target}")
    return {"deleted": body.folder_name, "path": target}


# ─── API 7: Liệt kê folder đã xử lý ─────────────────────────────────────────
@router.get(
    "/list-output-folders",
    summary="Liệt kê các folder đã có kết quả trong output folder"
)
async def list_processed_folders():
    """
    UI gọi khi người dùng chọn Output Folder để đồng bộ trạng thái "Done".
    Trả về list tên folder → UI so sánh với tên video để đánh dấu ✓ Done.

    **Response:**
    ```json
    { "processed_folders": ["cam1_20240101", "cam2_20240102"] }
    ```
    """
    if not OUTPUT_DIR.exists():
        return {"processed_folders": [], "count": 0}
    folders = list_output_subfolders(str(OUTPUT_DIR))
    return {"processed_folders": folders, "count": len(folders)}


# ─── API 8: Lấy tóm tắt tất cả jobs hiện tại ─────────────────────────────────
@router.get("/jobs", summary="Lấy danh sách tất cả jobs và trạng thái")
async def list_jobs():
    """Hữu ích để debug hoặc khôi phục UI sau khi reload trang."""
    return {
        "total": len(_jobs),
        "jobs":  _jobs,
    }
    
# ____API 9: dọn dẹp RAM các job đã xử lý xong
@router.post("/clear-done-job", summary="delete cache of old log in the Server to free up memory")
async def clear_done_jobs():
    keys_to_delete = []
    
    # tìm những jobs có trạng thái đã xử lý xong
    for key, job_data in _jobs.items():
        if job_data["status"] not in ["waiting", "processing"]:
            keys_to_delete.append(key)
            
    # xóa những jobs trên
    for k in keys_to_delete:
        _jobs.pop(k, None)
        
    return {"message": "đã làm sạch lịch sử server thành công",
            "clear count": len(keys_to_delete),
            "remaining_active": len(_jobs)}
    
# 
@router.get("/get/{video_name}/crops/{image_name}", summary="Send crop Image to user device")
async def get_crop_image(video_name: str, image_name: str):
    image_path = OUTPUT_DIR / video_name / "crops" / image_name
    
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="No found image")
    
    return FileResponse(image_path)