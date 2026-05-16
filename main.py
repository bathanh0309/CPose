"""
CPose — FastAPI backend (fixed + performance optimized)
"""

import asyncio
import os
import socket
import time
import uuid
import threading
from collections import deque
from pathlib import Path
from typing import Annotated, Dict, Optional, Set
from urllib.parse import urlparse

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;15000000|max_delay;500000"

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from src.core.ui_logger import ui_logger
from src.core.web_runtime import WebAIProcessor
from src.utils.config import load_pipeline_cfg

BASE_DIR   = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "data" / "config" / "resources.txt"
PIPELINE_CONFIG = BASE_DIR / "configs" / "system" / "pipeline.yaml"
UPLOAD_DIR  = BASE_DIR / "data" / "uploads"
OUTPUT_DIR  = BASE_DIR / "data" / "output" / "vis"
STATIC_DIR  = BASE_DIR / "static"

BUFFER_MAX   = 300
_frame_buffers: Dict[str, deque] = {}   
_session_meta: Dict[str, dict]   = {}   
_recording_flags: Dict[str, bool] = {}
_pipeline_cfg: Optional[dict] = None

app = FastAPI(title="CPose Control Panel")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_MODULES: Set[str] = {"pose", "track", "reid", "adl"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def parse_modules(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {m.strip().lower() for m in raw.split(",") if m.strip().lower() in VALID_MODULES}

def get_pipeline_cfg() -> dict:
    global _pipeline_cfg
    if _pipeline_cfg is None:
        _pipeline_cfg = load_pipeline_cfg(PIPELINE_CONFIG, BASE_DIR)
    return _pipeline_cfg

def format_terminal_ai_log(cam_id: str, msg: str) -> str:
    if ":" in msg:
        module, rest = msg.split(":", 1)
        return f"[cam={cam_id}][{module.strip()}] {rest.strip()}"
    return f"[cam={cam_id}] {msg}"

def mask_rtsp_credentials(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        return value

    host = parsed.hostname
    host_parts = host.split(".")
    if len(host_parts) == 4 and all(part.isdigit() for part in host_parts):
        host = ".".join([*host_parts[:3], "***"])
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = host
    if parsed.username or parsed.password:
        username = parsed.username or "username"
        netloc = f"{username}:***@{host}"
    return parsed._replace(netloc=netloc).geturl()

def camera_to_payload(camera: dict, index: int) -> dict:
    url_masked = mask_rtsp_credentials(camera["url"])
    name = camera["name"]
    return {
        "id": f"camera:{index}",
        "name": name,
        "url_masked": url_masked,
        "display": f"{name} — {url_masked}",
    }

async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        if websocket.client_state != WebSocketState.CONNECTED or websocket.application_state != WebSocketState.CONNECTED:
            return False
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False
    except Exception as exc:
        print(f"[WS] send warning: {exc}")
        return False

async def safe_send_bytes(websocket: WebSocket, payload: bytes) -> bool:
    try:
        if websocket.client_state != WebSocketState.CONNECTED or websocket.application_state != WebSocketState.CONNECTED:
            return False
        await websocket.send_bytes(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False
    except Exception as exc:
        print(f"[WS] send bytes warning: {exc}")
        return False

async def safe_close_websocket(websocket: WebSocket) -> None:
    try:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        pass

async def receive_commands(websocket: WebSocket, processor: WebAIProcessor, stop_event: asyncio.Event, log_queue: asyncio.Queue) -> None:
    while not stop_event.is_set():
        try:
            msg = await websocket.receive_json()
            if msg.get("type") == "set_modules":
                active_modules = parse_modules(msg.get("modules", ""))
                processor.set_modules(active_modules)
                await log_queue.put(("system", f"Modules updated: {', '.join(active_modules) or 'none'}"))
            elif msg.get("type") == "stop":
                stop_event.set()
                break
        except Exception:
            stop_event.set()
            break

def is_local_file_source(source: str) -> bool:
    if "://" in source:
        return False
    return Path(source).exists()

def read_camera_sources():
    cameras = []
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "__" not in line:
                    continue
                name, url = line.split("__", 1)
                name = name.strip()
                url = url.strip()
                if name and url.lower().startswith("rtsp://"):
                    cameras.append({"name": name, "url": url})
    return cameras

def resolve_video_source(source: Optional[str], url: Optional[str]) -> tuple[Optional[str], str]:
    if source and source.startswith("camera:"):
        try:
            camera_index = int(source.split(":", 1)[1])
        except ValueError:
            return None, "Invalid camera"
        cameras = read_camera_sources()
        if camera_index < 0 or camera_index >= len(cameras):
            return None, "Camera does not exist"
        return cameras[camera_index]["url"], cameras[camera_index]["name"]
    if source:
        return source, "File upload"
    if url:
        return url, "RTSP Source"
    return None, "No video source selected"

def describe_rtsp_tcp_status(video_source: str, timeout: float = 3.0) -> Optional[str]:
    parsed = urlparse(video_source)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        return None

    port = parsed.port or 554
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return None
    except OSError as exc:
        return f"RTSP TCP check failed: cannot connect to {mask_rtsp_credentials(video_source)} ({exc})"

def open_video_capture(video_source: str) -> cv2.VideoCapture:
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        15000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        15000,
    ]
    try:
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG, params)
    except Exception:
        cap = cv2.VideoCapture(video_source)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


class ThreadedCamera:
    """
    Non-blocking OpenCV camera reader.

    cv2.VideoCapture.read() blocks on slow RTSP/network sources. This class
    moves capture.read() into a background daemon thread and keeps only the
    latest frame. The async WebSocket loop reads the latest frame immediately.
    """

    def __init__(
        self,
        source: str,
        loop_local_video: bool = True,
        name: str = "camera",
        sleep_sec: float = 0.005,
        max_read_failures: int = 30,
        reconnect_delay: float = 2.0,
    ):
        self.source = source
        self.loop_local_video = bool(loop_local_video)
        self.name = name
        self.sleep_sec = float(sleep_sec)
        self.max_read_failures = max(1, int(max_read_failures))
        self.reconnect_delay = max(0.1, float(reconnect_delay))

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.last_frame_ts: float = 0.0

        self.width: int = 640
        self.height: int = 480
        self.fps: float = 25.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._opened_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._opened = False
        self._error: Optional[str] = None
        self._read_failures = 0

    def start(self) -> "ThreadedCamera":
        if self._thread and self._thread.is_alive():
            return self

        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"ThreadedCamera-{self.name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _release_capture(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        finally:
            self.cap = None

    def _open_capture(self) -> bool:
        self._release_capture()
        cap = open_video_capture(self.source)

        if not cap or not cap.isOpened():
            self._error = f"OpenCV cannot open source: {mask_rtsp_credentials(self.source)}"
            self._opened = False
            self._opened_event.set()
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            self._release_capture()
            return False

        self.cap = cap
        self._opened = True
        self._error = None
        self._read_failures = 0

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 1 or fps > 240:
            fps = 25.0

        self.fps = fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self._opened_event.set()
        print(f"[RTSP] Camera opened {self.name}: {mask_rtsp_credentials(self.source)}")
        return True

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    if not self._open_capture():
                        time.sleep(self.reconnect_delay)
                    continue

                ok, frame = self.cap.read()

                if not ok or frame is None:
                    if is_local_file_source(self.source) and self.loop_local_video:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(self.sleep_sec)
                        continue

                    self._read_failures += 1
                    self._error = (
                        f"Lost signal or cannot read frame "
                        f"({self._read_failures}/{self.max_read_failures}). Reconnecting..."
                    )
                    if self._read_failures >= self.max_read_failures:
                        print(f"[RTSP] Reconnecting {self.name}: {mask_rtsp_credentials(self.source)}")
                        self._opened = False
                        self._release_capture()
                        time.sleep(self.reconnect_delay)
                    else:
                        time.sleep(0.05)
                    continue

                with self._lock:
                    self.frame = frame
                    self.last_frame_ts = time.monotonic()

                self._opened = True
                self._error = None
                self._read_failures = 0

                if is_local_file_source(self.source):
                    time.sleep(max(self.sleep_sec, 1.0 / max(float(self.fps), 1.0)))
                else:
                    time.sleep(self.sleep_sec)

            except Exception as exc:
                self._error = f"ThreadedCamera error: {exc}. Reconnecting..."
                self._opened = False
                self._opened_event.set()
                print(f"[RTSP] {self.name} reader warning: {exc}")
                self._release_capture()
                time.sleep(self.reconnect_delay)

        self._release_capture()

    def wait_opened(self, timeout: float = 5.0) -> bool:
        self._opened_event.wait(timeout=timeout)
        return self._opened

    def read(self, copy: bool = True) -> Optional[np.ndarray]:
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy() if copy else self.frame

    def get_meta(self) -> dict:
        return {"fps": self.fps, "width": self.width, "height": self.height}

    def age(self) -> float:
        if self.last_frame_ts <= 0:
            return float("inf")
        return time.monotonic() - self.last_frame_ts

    def error(self) -> Optional[str]:
        return self._error

    def release(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()

        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=join_timeout)
        except RuntimeError:
            pass

        self._release_capture()

# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/style.css")
def stylesheet():
    content = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    return Response(content=content, media_type="text/css")

@app.get("/scripts.js")
def scripts():
    content = (STATIC_DIR / "scripts.js").read_text(encoding="utf-8")
    return Response(content=content, media_type="application/javascript")

# ---------------------------------------------------------------------------
# Routes — cameras & upload
# ---------------------------------------------------------------------------
@app.get("/api/cameras")
def get_cameras():
    cameras = [camera_to_payload(c, i) for i, c in enumerate(read_camera_sources())]
    return JSONResponse(content=cameras)

@app.post("/api/upload")
async def upload_video(file: Annotated[UploadFile, File(...)]):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "uploaded-video").name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in VALID_VIDEO_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported video extension: {suffix or '(none)'}"},
        )
    target = UPLOAD_DIR / safe_name
    stem, suffix, counter = target.stem, target.suffix, 1
    while target.exists():
        target = UPLOAD_DIR / f"{stem}-{counter}{suffix}"
        counter += 1
    with target.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"name": target.name, "source": str(target), "type": "uploaded_video"}

# ---------------------------------------------------------------------------
# Save-on-demand endpoint
# ---------------------------------------------------------------------------
@app.post("/api/save-video/{session_id}")
async def save_video(session_id: str):
    if session_id not in _frame_buffers or not _frame_buffers[session_id]:
        return JSONResponse(status_code=404, content={"error": "No frames in buffer"})

    meta = _session_meta.get(session_id, {})
    fps = meta.get("fps", 25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = OUTPUT_DIR / f"session_{session_id[:8]}.mp4"

    frames = list(_frame_buffers[session_id])
    first_frame = None
    for jpeg_bytes in frames:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        first_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if first_frame is not None:
            break
    if first_frame is None:
        return JSONResponse(status_code=404, content={"error": "No decodable frames in buffer"})

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_name), fourcc, fps, (width, height))

    for jpeg_bytes in frames:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            writer.write(frame)
    writer.release()

    return {"saved": out_name.name, "frames": len(frames), "path": str(out_name)}

@app.post("/api/recording/start/{session_id}")
async def start_recording(session_id: str):
    maxlen = int(get_pipeline_cfg().get("web", {}).get("recording_buffer_max", BUFFER_MAX))
    _frame_buffers[session_id] = deque(maxlen=maxlen)
    _recording_flags[session_id] = True
    return {"session_id": session_id, "recording": True, "buffer_max": maxlen}

@app.post("/api/recording/stop/{session_id}")
async def stop_recording(session_id: str):
    _recording_flags[session_id] = False
    return {"session_id": session_id, "recording": False, "buffered_frames": len(_frame_buffers.get(session_id, []))}

@app.post("/api/save-excel/{session_id}")
async def save_excel(session_id: str):
    return JSONResponse(status_code=501, content={"error": "Save Excel is not implemented yet", "session_id": session_id})

@app.get("/api/sessions")
def list_sessions():
    return {sid: {"buffered_frames": len(buf), **_session_meta.get(sid, {})} for sid, buf in _frame_buffers.items()}

@app.get("/api/logs/{camera_id}")
def get_logs(camera_id: str):
    return {"camera_id": camera_id, "logs": ui_logger.get_logs(camera_id)}

@app.get("/api/metrics/{camera_id}")
def get_metrics(camera_id: str):
    return {"camera_id": camera_id, "metrics": ui_logger.get_metrics(camera_id)}

@app.get("/api/status")
def get_status():
    return {
        "sessions": {sid: {"buffered_frames": len(buf), **_session_meta.get(sid, {})} for sid, buf in _frame_buffers.items()},
        "ui_logger": ui_logger.status(),
    }

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream/{cam_id}")
async def stream_video(
    websocket: WebSocket, cam_id: str, source: Optional[str] = None, url: Optional[str] = None,
    modules: Optional[str] = None, session_id: Optional[str] = None,
):
    await websocket.accept()
    connected = True
    camera: Optional[ThreadedCamera] = None

    active_modules = parse_modules(modules)
    cfg = get_pipeline_cfg()
    web_cfg = cfg.get("web", {})
    metrics_interval = int(web_cfg.get("metrics_interval_frames", cfg.get("ui", {}).get("metrics_interval_frames", 5)))
    target_fps = float(web_cfg.get("target_fps", 25))
    target_interval = 1.0 / max(target_fps, 1.0)
    jpeg_quality_base = int(web_cfg.get("jpeg_quality", 75))
    jpeg_quality_min = int(web_cfg.get("jpeg_quality_min", 50))
    jpeg_quality = jpeg_quality_base
    threaded_camera = bool(web_cfg.get("threaded_camera", True))
    stream_width = int(web_cfg.get("stream_width", 0) or 0)
    infer_every_n_frames = max(1, int(web_cfg.get("infer_every_n_frames", 1) or 1))
    loop_local_video = bool(web_cfg.get("loop_local_video", True))
    lag_threshold = float(web_cfg.get("frame_skip_lag_multiplier", 2.0)) * target_interval
    recording_buffer_max = int(web_cfg.get("recording_buffer_max", BUFFER_MAX))
    processor = WebAIProcessor(camera_id=cam_id, modules=active_modules, cfg=cfg)

    if not session_id:
        session_id = str(uuid.uuid4())

    _frame_buffers[session_id] = deque(maxlen=recording_buffer_max)
    _recording_flags[session_id] = False

    if not await safe_send_json(websocket, {"type": "session", "session_id": session_id, "active_modules": list(active_modules)}):
        return

    video_source, source_label = resolve_video_source(source, url)
    if not video_source:
        if await safe_send_json(websocket, {"type": "log", "msg": source_label, "level": "error"}):
            await safe_close_websocket(websocket)
        return

    rtsp_tcp_error = describe_rtsp_tcp_status(video_source)
    if rtsp_tcp_error:
        print(f"[RTSP] TCP check failed for {mask_rtsp_credentials(video_source)}")
        if await safe_send_json(websocket, {"type": "log", "msg": "RTSP endpoint unreachable. Please check camera/port in config.", "level": "error"}):
            await safe_close_websocket(websocket)
        return

    if not await safe_send_json(websocket, {"type": "log", "msg": f"Connected to: {source_label} | Modules: {', '.join(active_modules) or 'none'}", "level": "system"}):
        return

    if not threaded_camera:
        print("[System] web.threaded_camera=false ignored; non-blocking streaming requires ThreadedCamera.")

    camera = ThreadedCamera(
        source=video_source,
        loop_local_video=loop_local_video,
        name=f"cam-{cam_id}",
        sleep_sec=0.005,
    ).start()

    if not camera.wait_opened(timeout=5.0):
        err_msg = camera.error() or "OpenCV/FFmpeg cannot open source."
        print(f"[RTSP] {err_msg}")
        if not await safe_send_json(websocket, {
            "type": "log",
            "msg": f"Warning: camera is not ready yet. Waiting for reconnect... {err_msg}",
            "level": "warning",
        }):
            camera.release()
            return

    meta = camera.get_meta()
    _session_meta[session_id] = {
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
    }

    stop_event = asyncio.Event()
    log_queue: asyncio.Queue = asyncio.Queue()
    command_task = asyncio.create_task(receive_commands(websocket, processor, stop_event, log_queue))
    last_loop_elapsed = 0.0
    last_stale_log_ts = 0.0

    try:
        while connected and not stop_event.is_set():
            frame_start = time.monotonic()

            frame = camera.read(copy=True)
            if frame is None:
                now = time.monotonic()
                if now - last_stale_log_ts >= 3.0:
                    last_stale_log_ts = now
                    msg = camera.error() or "Camera stream is stale. Waiting for reconnect..."
                    if not await safe_send_json(websocket, {"type": "log", "msg": msg, "level": "warning"}):
                        connected = False
                        break
                await asyncio.sleep(0.01)
                continue

            if camera.age() > 3.0:
                now = time.monotonic()
                if now - last_stale_log_ts >= 3.0:
                    last_stale_log_ts = now
                    if not await safe_send_json(websocket, {
                        "type": "log",
                        "msg": "Camera stream is stale. Waiting for reconnect...",
                        "level": "warning",
                    }):
                        connected = False
                        break
                await asyncio.sleep(0.02)
                continue

            while not log_queue.empty():
                level, msg = await log_queue.get()
                if not await safe_send_json(websocket, {"type": "log", "msg": msg, "level": level}):
                    connected = False
                    break
            if not connected:
                break

            next_frame_idx = processor.frame_idx + 1
            skip_for_interval = bool(processor.modules) and (next_frame_idx % infer_every_n_frames != 0)
            skip_for_lag = bool(processor.modules) and last_loop_elapsed > lag_threshold
            skip_ai = skip_for_interval or skip_for_lag
            if skip_ai:
                processor.frame_idx += 1
                fps_actual = round(1.0 / max(last_loop_elapsed, 1e-6), 2)
                metrics = {
                    "camera_id": cam_id, "frame_idx": processor.frame_idx, "fps": fps_actual,
                    "modules": sorted(processor.modules), "detections": 0, "tracked": 0, "filtered": 0,
                    "skip_ai": True, "skip_reason": "lag" if skip_for_lag else "interval",
                }
                ai_logs = []
            else:
                frame, ai_logs, metrics = processor.process(frame)

            for level, log_msg in ai_logs:
                print(format_terminal_ai_log(cam_id, log_msg))
                if not await safe_send_json(websocket, {"type": "log", "msg": log_msg, "level": level}):
                    connected = False
                    break
            if not connected:
                break

            if metrics_interval > 0 and processor.frame_idx % metrics_interval == 0:
                if not await safe_send_json(websocket, {"type": "metric", "metrics": metrics}):
                    break

            # Send gallery crop events
            for event in getattr(processor, "gallery_events", []):
                if not await safe_send_json(websocket, event):
                    connected = False
                    break
            if not connected:
                break

            fps_actual = float(metrics.get("fps", 0.0) or 0.0)
            if fps_actual and fps_actual < target_fps * 0.6:
                jpeg_quality = jpeg_quality_min
            elif fps_actual > target_fps * 0.9:
                jpeg_quality = jpeg_quality_base

            if stream_width > 0:
                h, w = frame.shape[:2]
                if w > stream_width:
                    scale = stream_width / float(w)
                    frame = cv2.resize(frame, (stream_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                elapsed = time.monotonic() - frame_start
                last_loop_elapsed = elapsed
                await asyncio.sleep(max(0.0, target_interval - elapsed))
                continue

            jpeg_bytes = buffer.tobytes()

            if _recording_flags.get(session_id, False):
                _frame_buffers[session_id].append(jpeg_bytes)

            if not await safe_send_bytes(websocket, jpeg_bytes):
                break

            elapsed = time.monotonic() - frame_start
            last_loop_elapsed = elapsed
            await asyncio.sleep(max(0.0, target_interval - elapsed))

    except WebSocketDisconnect:
        print(f"[System] Client disconnected: {cam_id}")
    finally:
        stop_event.set()
        command_task.cancel()
        try:
            await command_task
        except Exception:
            pass
        try:
            if camera is not None:
                camera.release()
        except Exception as exc:
            print(f"[System] camera.release warning cam={cam_id}: {exc}")
        print(f"[System] Stream closed cam={cam_id}, session={session_id}")
