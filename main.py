"""
CPose â€” FastAPI backend (fixed)

Fixes applied:
  #1/#3  Module routing: WS nháº­n query param ?modules=pose,track,reid,adl
  #2/#5  Save-on-demand: POST /api/save-video/{cam_id} ghi buffer ra file
  #4     XÃ³a fake log (frame % 90)
  #6     Pipeline hook sáºµn sÃ ng uncomment khi model weights cÃ³ máº·t
  #7     Frame buffer per session (deque) Ä‘á»ƒ save-on-demand hoáº¡t Ä‘á»™ng
"""

# Performance fixes:
# FIX #8  Adaptive sleep theo target_fps.
# FIX #9  Binary WebSocket JPEG frames, JSON chỉ cho control/log/metric.
# FIX #10 Receive command task riêng, hot loop không wait_for.
# FIX #11 Skip AI overlay khi loop lag, vẫn gửi frame raw.
# FIX #12 Recording buffer 300 frame, opt-in qua recording flag.
# FIX #13 POST /api/recording/start|stop/{session_id}.
# FIX #14 CAP_PROP_BUFFERSIZE=1 cho OpenCV capture.
# FIX #15 JPEG quality adaptive 75 -> 50 -> 75.
# FIX #16 Frontend render ArrayBuffer binary frames.
# FIX #17 Frontend FPS badge.
# FIX #18 Frontend stale-frame overlay.

import asyncio
import socket
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Annotated, Dict, Optional, Set
from urllib.parse import urlparse

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

# ---------------------------------------------------------------------------
# Frame buffer: session_id â†’ deque of encoded JPEG bytes (giá»¯ tá»‘i Ä‘a 9000 frame ~ 5 phÃºt @30fps)
# ---------------------------------------------------------------------------
BUFFER_MAX   = 300
_frame_buffers: Dict[str, deque] = {}   # session_id â†’ deque[bytes]
_session_meta: Dict[str, dict]   = {}   # session_id â†’ {width, height, fps}
_recording_flags: Dict[str, bool] = {}
_pipeline_cfg: Optional[dict] = None

# ---------------------------------------------------------------------------
# (Optional) Pipeline import â€” uncomment khi model weights Ä‘Ã£ sáºµn sÃ ng
# ---------------------------------------------------------------------------
# from src.core.pipeline import CPosePipeline
# _pipeline: Optional[CPosePipeline] = None

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


def parse_modules(raw: Optional[str]) -> Set[str]:
    """'pose,track,reid' â†’ {'pose', 'track', 'reid'}"""
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
        "display": f"{name} â€” {url_masked}",
    }


async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    """
    Send JSON only while the websocket is connected.
    Return False when the client disconnected or a close was already sent.
    """
    try:
        if (
            websocket.client_state != WebSocketState.CONNECTED
            or websocket.application_state != WebSocketState.CONNECTED
        ):
            return False
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if "Cannot call" not in str(exc) and "close message" not in str(exc):
            print(f"[WS] send runtime warning: {exc}")
        return False
    except Exception as exc:
        print(f"[WS] send warning: {exc}")
        return False


async def safe_send_bytes(websocket: WebSocket, payload: bytes) -> bool:
    try:
        if (
            websocket.client_state != WebSocketState.CONNECTED
            or websocket.application_state != WebSocketState.CONNECTED
        ):
            return False
        await websocket.send_bytes(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if "Cannot call" not in str(exc) and "close message" not in str(exc):
            print(f"[WS] send bytes runtime warning: {exc}")
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


async def receive_commands(
    websocket: WebSocket,
    processor: WebAIProcessor,
    stop_event: asyncio.Event,
    log_queue: asyncio.Queue,
) -> None:
    while not stop_event.is_set():
        try:
            msg = await websocket.receive_json()
            if msg.get("type") == "set_modules":
                active_modules = parse_modules(msg.get("modules", ""))
                processor.set_modules(active_modules)
                await log_queue.put((
                    "system",
                    f"Modules cáº­p nháº­t: {', '.join(active_modules) or 'none'}",
                ))
            elif msg.get("type") == "stop":
                stop_event.set()
                break
        except WebSocketDisconnect:
            stop_event.set()
            break
        except RuntimeError:
            stop_event.set()
            break
        except Exception as exc:
            print(f"[WS] receive warning cam={processor.camera_id}: {exc}")


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
            return None, "Camera khÃ´ng há»£p lá»‡"
        cameras = read_camera_sources()
        if camera_index < 0 or camera_index >= len(cameras):
            return None, "Camera khÃ´ng tá»“n táº¡i"
        return cameras[camera_index]["url"], cameras[camera_index]["name"]
    if source:
        return source, "File upload"
    if url:
        return url, "Nguá»“n RTSP"
    return None, "ChÆ°a chá»n nguá»“n video"


def describe_rtsp_tcp_status(video_source: str, timeout: float = 3.0) -> Optional[str]:
    parsed = urlparse(video_source)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        return None

    port = parsed.port or 554
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return None
    except OSError as exc:
        return (
            f"RTSP TCP check failed: cannot connect to "
            f"{mask_rtsp_credentials(video_source)} ({exc})"
        )


def open_video_capture(video_source: str) -> cv2.VideoCapture:
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        5000,
    ]
    try:
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG, params)
    except Exception:
        cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------------------------
# Routes â€” static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/style.css")
def stylesheet():
    content = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/css",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/scripts.js")
def scripts():
    content = (STATIC_DIR / "scripts.js").read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

# ---------------------------------------------------------------------------
# Routes â€” cameras & upload
# ---------------------------------------------------------------------------

@app.get("/api/cameras")
def get_cameras():
    cameras = [camera_to_payload(c, i) for i, c in enumerate(read_camera_sources())]
    return JSONResponse(content=cameras)


@app.post("/api/cameras/config")
async def upload_camera_config(file: Annotated[UploadFile, File(...)]):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        return JSONResponse(status_code=400, content={"error": "Only .txt camera config files are supported"})

    content = (await file.read()).decode("utf-8-sig")
    cameras = []
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "__" not in line:
            return JSONResponse(status_code=400, content={"error": f"Invalid camera line: {line}"})

        name, url = line.split("__", 1)
        name = name.strip()
        url = url.strip()
        if not name:
            return JSONResponse(status_code=400, content={"error": f"Invalid camera line: {line}"})
        if not url.lower().startswith("rtsp://"):
            return JSONResponse(status_code=400, content={"error": f"Invalid camera line: {line}"})

        cameras.append({"name": name, "url": url})
        lines.append(f"{name}__{url}")

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"cameras": [camera_to_payload(camera, i) for i, camera in enumerate(cameras)]}


@app.post("/api/upload")
async def upload_video(file: Annotated[UploadFile, File(...)]):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "uploaded-video").name
    target = UPLOAD_DIR / safe_name
    stem, suffix, counter = target.stem, target.suffix, 1
    while target.exists():
        target = UPLOAD_DIR / f"{stem}-{counter}{suffix}"
        counter += 1
    with target.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"name": target.name, "source": str(target)}


# ---------------------------------------------------------------------------
# FIX #2 / #5 â€” Save-on-demand endpoint
# ---------------------------------------------------------------------------

@app.post("/api/save-video/{session_id}")
async def save_video(session_id: str):
    """
    Ghi frame buffer cá»§a session ra file MP4.
    Frontend gá»i endpoint nÃ y khi ngÆ°á»i dÃ¹ng nháº¥n nÃºt "LÆ°u video".
    KhÃ´ng cÃ³ auto-save nÃ o xáº£y ra náº¿u khÃ´ng gá»i endpoint nÃ y.
    """
    if session_id not in _frame_buffers or not _frame_buffers[session_id]:
        return JSONResponse(status_code=404, content={"error": "KhÃ´ng cÃ³ frame nÃ o trong buffer"})

    meta = _session_meta.get(session_id, {})
    fps    = meta.get("fps", 25)
    width  = meta.get("width", 640)
    height = meta.get("height", 480)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = OUTPUT_DIR / f"session_{session_id[:8]}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_name), fourcc, fps, (width, height))

    frames = list(_frame_buffers[session_id])
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
    return {
        "session_id": session_id,
        "recording": False,
        "buffered_frames": len(_frame_buffers.get(session_id, [])),
    }


@app.post("/api/save-excel/{session_id}")
async def save_excel(session_id: str):
    return JSONResponse(
        status_code=501,
        content={"error": "Save Excel is not implemented yet", "session_id": session_id},
    )


@app.get("/api/sessions")
def list_sessions():
    return {
        sid: {
            "buffered_frames": len(buf),
            **_session_meta.get(sid, {}),
        }
        for sid, buf in _frame_buffers.items()
    }


@app.get("/api/logs/{camera_id}")
def get_logs(camera_id: str):
    return {"camera_id": camera_id, "logs": ui_logger.get_logs(camera_id)}


@app.get("/api/metrics/{camera_id}")
def get_metrics(camera_id: str):
    return {"camera_id": camera_id, "metrics": ui_logger.get_metrics(camera_id)}


@app.get("/api/status")
def get_status():
    return {
        "sessions": {
            sid: {
                "buffered_frames": len(buf),
                **_session_meta.get(sid, {}),
            }
            for sid, buf in _frame_buffers.items()
        },
        "ui_logger": ui_logger.status(),
    }


# ---------------------------------------------------------------------------
# FIX #1 / #3 â€” WebSocket vá»›i module routing + frame buffer
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream/{cam_id}")
async def stream_video(
    websocket: WebSocket,
    cam_id: str,
    source: Optional[str] = None,
    url: Optional[str] = None,
    modules: Optional[str] = None,   # â† FIX #3: nháº­n danh sÃ¡ch module tá»« UI
    session_id: Optional[str] = None,
):
    await websocket.accept()
    connected = True

    # Parse modules
    active_modules = parse_modules(modules)
    cfg = get_pipeline_cfg()
    web_cfg = cfg.get("web", {})
    metrics_interval = int(web_cfg.get("metrics_interval_frames", cfg.get("ui", {}).get("metrics_interval_frames", 5)))
    target_fps = float(web_cfg.get("target_fps", 25))
    target_interval = 1.0 / max(target_fps, 1.0)
    jpeg_quality_base = int(web_cfg.get("jpeg_quality", 75))
    jpeg_quality_min = int(web_cfg.get("jpeg_quality_min", 50))
    jpeg_quality = jpeg_quality_base
    loop_local_video = bool(web_cfg.get("loop_local_video", True))
    lag_threshold = float(web_cfg.get("frame_skip_lag_multiplier", 2.0)) * target_interval
    recording_buffer_max = int(web_cfg.get("recording_buffer_max", BUFFER_MAX))
    processor = WebAIProcessor(camera_id=cam_id, modules=active_modules, cfg=cfg)

    # Táº¡o session_id náº¿u chÆ°a cÃ³
    if not session_id:
        session_id = str(uuid.uuid4())

    # Khá»Ÿi táº¡o buffer cho session nÃ y
    _frame_buffers[session_id] = deque(maxlen=recording_buffer_max)
    _recording_flags[session_id] = False

    # Gá»­i session_id vá» client Ä‘á»ƒ sau nÃ y gá»i /api/save-video/{session_id}
    if not await safe_send_json(websocket, {
        "type": "session",
        "session_id": session_id,
        "active_modules": list(active_modules),
    }):
        return

    video_source, source_label = resolve_video_source(source, url)
    if not video_source:
        if await safe_send_json(websocket, {"type": "log", "msg": source_label, "level": "error"}):
            await safe_close_websocket(websocket)
        return

    rtsp_tcp_error = describe_rtsp_tcp_status(video_source)
    if rtsp_tcp_error:
        print(f"[RTSP] TCP check failed for {mask_rtsp_credentials(video_source)}")
        if await safe_send_json(websocket, {
            "type": "log",
            "msg": "RTSP endpoint unreachable. Kiem tra camera/port trong config.",
            "level": "error",
        }):
            await safe_close_websocket(websocket)
        return

    if not await safe_send_json(websocket, {
        "type": "log",
        "msg": f"Káº¿t ná»‘i: {source_label} | Modules: {', '.join(active_modules) or 'none'}",
        "level": "system",
    }):
        return

    cap = open_video_capture(video_source)
    if not cap.isOpened():
        print(f"[RTSP] OpenCV cannot open video source: {mask_rtsp_credentials(video_source)}")
        if await safe_send_json(websocket, {
            "type": "log",
            "msg": f"Lá»—i: KhÃ´ng thá»ƒ má»Ÿ nguá»“n video {source_label}. OpenCV/FFmpeg cannot open source.",
            "level": "error",
        }):
            await safe_close_websocket(websocket)
        return

    # LÆ°u metadata cho session (dÃ¹ng khi save)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _session_meta[session_id] = {"fps": fps, "width": width, "height": height}

    stop_event = asyncio.Event()
    log_queue: asyncio.Queue = asyncio.Queue()
    command_task = asyncio.create_task(receive_commands(websocket, processor, stop_event, log_queue))
    last_loop_elapsed = 0.0

    try:
        while connected and not stop_event.is_set():
            frame_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                if is_local_file_source(video_source) and loop_local_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if not await safe_send_json(websocket, {
                    "type": "log",
                    "msg": "Mat tin hieu hoac khong doc duoc frame.",
                    "level": "error",
                }):
                    connected = False
                break

            while not log_queue.empty():
                level, msg = await log_queue.get()
                if not await safe_send_json(websocket, {"type": "log", "msg": msg, "level": level}):
                    connected = False
                    break
            if not connected:
                break

            skip_ai = bool(processor.modules) and last_loop_elapsed > lag_threshold
            if skip_ai:
                processor.frame_idx += 1
                fps_actual = round(1.0 / max(last_loop_elapsed, 1e-6), 2)
                metrics = {
                    "camera_id": cam_id,
                    "frame_idx": processor.frame_idx,
                    "fps": fps_actual,
                    "modules": sorted(processor.modules),
                    "detections": 0,
                    "tracked": 0,
                    "filtered": 0,
                    "skip_ai": True,
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

            fps_actual = float(metrics.get("fps", 0.0) or 0.0)
            if fps_actual and fps_actual < target_fps * 0.6:
                jpeg_quality = jpeg_quality_min
            elif fps_actual > target_fps * 0.9:
                jpeg_quality = jpeg_quality_base

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
        print(f"[System] Client ngat ket noi {cam_id}")
    finally:
        stop_event.set()
        command_task.cancel()
        try:
            await command_task
        except (asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
            pass
        try:
            cap.release()
        except Exception as exc:
            print(f"[System] cap.release warning cam={cam_id}: {exc}")
        print(f"[System] Stream closed cam={cam_id}, session={session_id}")
