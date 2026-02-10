"""
HAVEN Web Server
FastAPI backend for Web UI: Registration + Live Stream + Dashboard
"""
import sys
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import time
import cv2
import numpy as np

# Paths
SRC_DIR = Path(__file__).parent
BACKEND_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from run import SequentialRunner

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="HAVEN Multi-Camera System", version="1.0")

# Templates
templates = Jinja2Templates(directory=str(SRC_DIR / "templates"))

# Global runner instance
runner: Optional[SequentialRunner] = None
runner_thread: Optional[threading.Thread] = None


# ============================================================
# Models
# ============================================================
class RegistrationRequest(BaseModel):
    name: str
    age: int
    gender: str  # "male" or "female"


# ============================================================
# Routes
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def start_system(reg: RegistrationRequest):
    """Start the HAVEN system with registration info."""
    global runner, runner_thread
    
    # Stop if already running
    if runner and runner._running:
        runner.stop()
        if runner_thread:
            runner_thread.join(timeout=10)
    
    # Create fresh runner (headless mode)
    runner = SequentialRunner(headless=True)
    
    # Set the profile from web form
    profiles = [{
        "name": reg.name,
        "age": reg.age,
        "gender": reg.gender
    }]
    runner.set_profiles(profiles)
    
    # Run in background thread
    def _run():
        try:
            runner.run(headless_override=True)
        except Exception as e:
            print(f"Runner error: {e}")
    
    runner_thread = threading.Thread(target=_run, daemon=True)
    runner_thread.start()
    
    return {"status": "started", "profile": profiles[0]}


@app.post("/api/stop")
async def stop_system():
    """Stop the running system."""
    global runner
    if runner and runner._running:
        runner.stop()
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    global runner
    if runner:
        status = runner.get_status()
        # Convert numpy/non-serializable types
        profiles = {}
        for k, v in status.get("profiles", {}).items():
            profiles[str(k)] = v
        status["profiles"] = profiles
        return JSONResponse(content=status)
    return JSONResponse(content={
        "running": False,
        "camera": "",
        "progress": {"current": 0, "total": 0, "camera": ""},
        "profiles": {},
        "logs": []
    })


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream."""
    def generate():
        while True:
            if runner:
                frame_bytes = runner.get_latest_frame()
                if frame_bytes:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                else:
                    # No frame yet — send blank
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for stream...", (400, 360),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
                    _, jpeg = cv2.imencode('.jpg', blank)
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                    )
            else:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "HAVEN - Ready", (480, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                )
            time.sleep(0.033)  # ~30fps
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
