import asyncio
import cv2
import logging
import queue
import shutil
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
_REPO_ROOT = Path(__file__).resolve().parents[4]
_POSE_ADL_SRC_DIR = _REPO_ROOT / "feat-pose-adl" / "backend" / "src"
if str(_POSE_ADL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_POSE_ADL_SRC_DIR))

from pose_adl_processor import PoseAdlClipProcessor

# App State
app = FastAPI(title="Pose ADL GUI Engine")

RAW_VIDEOS_DIR = _REPO_ROOT / "data" / "raw_videos"
PROCESSED_VIDEOS_DIR = _REPO_ROOT / "data" / "processed_videos"
HTML_PATH = Path(__file__).parent / "pose_adl_gui.html"

RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# We serve the processed videos dir so frontend can play/download them
app.mount("/processed", StaticFiles(directory=str(PROCESSED_VIDEOS_DIR)), name="processed")


class State:
    def __init__(self):
        self.status = "idle"
        self.error = ""
        self.frame_idx = 0
        self.total_frames = 0
        self.posture_counts = {}
        self.output_filename = ""
        self.output_url = ""
        self.frame_queue = queue.Queue(maxsize=10)

state = State()


def generate_mjpeg():
    # Keep streaming frames from the queue
    while True:
        if state.status == "completed" or state.status == "error":
            break
        try:
            # wait briefly to not lock up when queue is empty
            frame = state.frame_queue.get(timeout=0.1)
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"MJPEG stream error: {e}")
            break


def frame_callback(annotated_frame, frame_idx, total_frames, posture_counts):
    state.frame_idx = frame_idx
    state.total_frames = total_frames
    state.posture_counts = posture_counts
    # Put frame in queue if not full. If full, drop frame (keeps live view real-time without memory bloat)
    try:
        state.frame_queue.put_nowait(annotated_frame)
    except queue.Full:
        pass


def run_processing(input_path: Path, output_path: Path):
    processor = PoseAdlClipProcessor()
    try:
        processor.process_video(
            input_path=input_path,
            output_path=output_path,
            frame_callback=frame_callback
        )
        state.status = "completed"
        state.output_url = f"/processed/{output_path.name}"
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        logging.exception("Processing failed")


@app.get("/", response_class=HTMLResponse)
async def serve_gui():
    return HTML_PATH.read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Reset state
    global state
    state = State()
    state.status = "processing"
    
    file_name = file.filename
    input_path = RAW_VIDEOS_DIR / file_name
    
    # Prefix processed name to ensure uniqueness or distinction
    out_name = f"proc_{int(time.time())}_{file_name}"
    output_path = PROCESSED_VIDEOS_DIR / out_name
    
    state.output_filename = out_name
    
    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Start thread
    threading.Thread(target=run_processing, args=(input_path, output_path), daemon=True).start()
    
    return {"message": "Started", "filename": file_name}


@app.get("/api/stream")
async def stream_video():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
async def get_status():
    return {
        "status": state.status,
        "error": state.error,
        "frame_idx": state.frame_idx,
        "total_frames": state.total_frames,
        "posture_counts": state.posture_counts,
        "output_filename": state.output_filename,
        "output_url": state.output_url
    }

def main():
    # Automatically open browser
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:8543/")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8543)

if __name__ == "__main__":
    main()
