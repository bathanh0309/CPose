# Tách frame, chạy AI model lớn, tạo labels
import cv2
import json
import asyncio
import logging

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from app.utils.file_handler import is_valid_mp4

logger = logging.getLogger(__name__)

# Config for detection
_model = YOLO("models/yolo11l.pt")

FRAME_SKIP = 15

# STOP FLAG 
_stop_flags: dict[str, bool] = {}

def request_stop(job_id: str):
    _stop_flags[job_id] = True
    

def clear_stop_flag(job_id: str):
    _stop_flags.pop(job_id, None)
    
    
def is_stop_requested(job_id: str) -> bool:
    return _stop_flags.get(job_id, False)

# Main functions
def extract_human_detection(video_path: str, output_folder: str, job_id: str, progress_cb: Optional[callable]) -> dict:
    
    clear_stop_flag(job_id)
    
    if not is_valid_mp4(video_path):
        raise ValueError(f"Not valid MP4 file {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Can not open video: {video_path}")
    
    # create output folder for the processing video
    # out_dir = ensure_output_subdir(output_folder, Path(video_path).name)
    original_stem = Path(video_path).name.split('_',1)[-1].replace('.mp4','')
    out_dir = Path(output_folder) / original_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = Path(out_dir) / "detections.txt"
    
    frame_dir = Path(out_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok = True)
    
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # detections: list[dict] = []
    txt_lines = []
    frame_idx:  int = 0
    processed_frames: int = 0
    last_reported_pct: int = 0
    detections = 0
    
    logger.info(f"[{job_id}] Start processing: {Path(video_path).name} ({total_frames} frames)")
    try:
        while True:
            # ── Kiểm tra stop flag trước mỗi frame ──────────────────────────
            if is_stop_requested(job_id):
                logger.info(f"[{job_id}] Dừng theo yêu cầu tại frame {frame_idx}")
                break

            ret, frame = cap.read()
            
            if not ret:
                break  # Hết video

            frame_idx += 1
            
            # ── Chỉ xử lý mỗi FRAME_SKIP frame ─────────────────────────────
            if frame_idx % FRAME_SKIP == 0:
                humans = _detect_humans(frame)
                processed_frames += 1

                if humans:
                    
                    valid_humans = []
                                        
                    for h in humans:
                        x1, y1, x2, y2 = h["x1"], h["y1"], h["x2"], h["y2"]

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = max(0, x2)
                        y2 = max(0, y2)

                        if x2 > x1 and y2 > y1:
                            valid_humans.append({
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            })                    
                                        
                    if valid_humans:

                        txt_lines.append(f"Frame {frame_idx} {len(valid_humans)}")
                        
                        for h in valid_humans:
                            txt_lines.append(
                                f"{h['x1']} {h['y1']} {h['x2']} {h['y2']}"
                            )
                        
                        txt_lines.append("")
                                                
                        frame_filename = f"frame_{frame_idx:05d}.jpg"
                        frame_path = frame_dir / frame_filename

                        cv2.imwrite(str(frame_path), frame)
                    
                        detections += 1
            
            # ── Callback tiến trình (mỗi 5%) ────────────────────────────────
            if progress_cb and total_frames > 0:
                pct = int((frame_idx / total_frames) * 100)
                if pct >= last_reported_pct + 5:
                    last_reported_pct = pct
                    try:
                        progress_cb(pct)
                    except Exception:
                        pass  # Không để lỗi callback làm hỏng luồng chính

    finally:
        cap.release()
        clear_stop_flag(job_id)

    # ── Lưu kết quả ra JSON ──────────────────────────────────────────────────
    txt_path = Path(out_dir) / "detections.txt"

    try:
        with open(txt_path, "w", encoding="utf-8") as f:

            # tên video
            f.write(f"{Path(video_path).name}\n")

            # số frame detect
            f.write(f"{detections}\n\n")

            for line in txt_lines:
                f.write(line + "\n")
    except OSError as e:
        raise RuntimeError(f"Không ghi được file kết quả: {e}") from e

    logger.info(
        f"[{job_id}] Hoàn tất. {processed_frames} frames có người / "
        f"{detections} frames đã xử lý. Output: {txt_path}"
    )

    return {
        "output_dir":        str(out_dir),
        "txt_path":         str(txt_path),
        "total_frames":      total_frames,
        "processed_frames":  processed_frames,
    }


def get_video_metadata(video_path: str) -> dict:
    # extract basic info of mp4 files
    if not is_valid_mp4(video_path):
        raise ValueError(f"Not valid mp4 file {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Can not open video: {video_path}")
    
    meta = {
        "filename": Path(video_path).name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS), 2)
    }
    
    cap.release()
    return meta


def _detect_humans(frame) -> list[dict]:
    results = _model(frame, classes=[0], verbose=False)
    humans = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        
        if conf > 0.3:
            humans.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": round(conf, 3),
            })
            
    return humans

# ASYNC WRAPPER for running FastAPI in background
async def extract_human_detection_async(
    video_path: str,
    output_folder: str,
    job_id: str,
    progress_cb: Optional[callable] = None,
) -> dict:
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: extract_human_detection(video_path, output_folder, job_id, progress_cb),
    )