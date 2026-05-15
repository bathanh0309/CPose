"""
CPose — FastAPI backend (fixed)

Fixes applied:
  #1/#3  Module routing: WS nhận query param ?modules=pose,track,reid,adl
  #2/#5  Save-on-demand: POST /api/save-video/{cam_id} ghi buffer ra file
  #4     Xóa fake log (frame % 90)
  #6     Pipeline hook sẵn sàng uncomment khi model weights có mặt
  #7     Frame buffer per session (deque) để save-on-demand hoạt động
"""

import asyncio
import base64
import socket
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

BASE_DIR   = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "data" / "config" / "resources.txt"
UPLOAD_DIR  = BASE_DIR / "data" / "uploads"
OUTPUT_DIR  = BASE_DIR / "data" / "output" / "vis"
STATIC_DIR  = BASE_DIR / "static"

# ---------------------------------------------------------------------------
# Frame buffer: session_id → deque of encoded JPEG bytes (giữ tối đa 9000 frame ~ 5 phút @30fps)
# ---------------------------------------------------------------------------
BUFFER_MAX   = 9000
_frame_buffers: Dict[str, deque] = {}   # session_id → deque[bytes]
_session_meta: Dict[str, dict]   = {}   # session_id → {width, height, fps}

# ---------------------------------------------------------------------------
# (Optional) Pipeline import — uncomment khi model weights đã sẵn sàng
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
    """'pose,track,reid' → {'pose', 'track', 'reid'}"""
    if not raw:
        return set()
    return {m.strip().lower() for m in raw.split(",") if m.strip().lower() in VALID_MODULES}


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


async def safe_close_websocket(websocket: WebSocket) -> None:
    try:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        pass


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
            return None, "Camera không hợp lệ"
        cameras = read_camera_sources()
        if camera_index < 0 or camera_index >= len(cameras):
            return None, "Camera không tồn tại"
        return cameras[camera_index]["url"], cameras[camera_index]["name"]
    if source:
        return source, "File upload"
    if url:
        return url, "Nguồn RTSP"
    return None, "Chưa chọn nguồn video"


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
        return cv2.VideoCapture(video_source, cv2.CAP_FFMPEG, params)
    except Exception:
        return cv2.VideoCapture(video_source)


def process_frame_with_modules(frame: np.ndarray, modules: Set[str]) -> tuple[np.ndarray, list[str]]:
    """
    Hook chạy pipeline theo modules được bật.
    Hiện tại trả nguyên frame + log trống.
    Uncomment từng khối khi model weights sẵn sàng.
    """
    logs: list[str] = []

    # ── Pose detection ──────────────────────────────────────────────────────
    if "pose" in modules:
        # from src.detectors.yolo_pose import YoloPoseTracker
        # results = pose_tracker.run(frame)
        # frame = draw_pose(frame, results)
        # logs.append(f"[POSE] {len(results)} người phát hiện")
        pass

    # ── ByteTrack ────────────────────────────────────────────────────────────
    if "track" in modules:
        # tracks = byte_tracker.update(results)
        # frame = draw_tracks(frame, tracks)
        # logs.append(f"[TRACK] {len(tracks)} track đang theo dõi")
        pass

    # ── FastReID ─────────────────────────────────────────────────────────────
    if "reid" in modules:
        # embeddings = reid.extract(frame, tracks)
        # person_ids = gallery.query(embeddings)
        # frame = draw_reid(frame, person_ids)
        # logs.append(f"[ReID] Gán ID: {person_ids}")
        pass

    # ── PoseC3D / ADL ────────────────────────────────────────────────────────
    if "adl" in modules:
        # action = posec3d.classify(pose_buffer)
        # frame = draw_adl(frame, action)
        # logs.append(f"[ADL] Hành động: {action}")
        pass

    return frame, logs


# ---------------------------------------------------------------------------
# Routes — static
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
# Routes — cameras & upload
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
# FIX #2 / #5 — Save-on-demand endpoint
# ---------------------------------------------------------------------------

@app.post("/api/save-video/{session_id}")
async def save_video(session_id: str):
    """
    Ghi frame buffer của session ra file MP4.
    Frontend gọi endpoint này khi người dùng nhấn nút "Lưu video".
    Không có auto-save nào xảy ra nếu không gọi endpoint này.
    """
    if session_id not in _frame_buffers or not _frame_buffers[session_id]:
        return JSONResponse(status_code=404, content={"error": "Không có frame nào trong buffer"})

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
# FIX #1 / #3 — WebSocket với module routing + frame buffer
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream/{cam_id}")
async def stream_video(
    websocket: WebSocket,
    cam_id: str,
    source: Optional[str] = None,
    url: Optional[str] = None,
    modules: Optional[str] = None,   # ← FIX #3: nhận danh sách module từ UI
    session_id: Optional[str] = None,
):
    await websocket.accept()
    connected = True

    # Parse modules
    active_modules = parse_modules(modules)

    # Tạo session_id nếu chưa có
    if not session_id:
        session_id = str(uuid.uuid4())

    # Khởi tạo buffer cho session này
    _frame_buffers[session_id] = deque(maxlen=BUFFER_MAX)

    # Gửi session_id về client để sau này gọi /api/save-video/{session_id}
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
        "msg": f"Kết nối: {source_label} | Modules: {', '.join(active_modules) or 'none'}",
        "level": "system",
    }):
        return

    cap = open_video_capture(video_source)
    if not cap.isOpened():
        print(f"[RTSP] OpenCV cannot open video source: {mask_rtsp_credentials(video_source)}")
        if await safe_send_json(websocket, {
            "type": "log",
            "msg": f"Lỗi: Không thể mở nguồn video {source_label}. OpenCV/FFmpeg cannot open source.",
            "level": "error",
        }):
            await safe_close_websocket(websocket)
        return

    # Lưu metadata cho session (dùng khi save)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _session_meta[session_id] = {"fps": fps, "width": width, "height": height}

    try:
        while connected:
            # Cho phép client gửi lệnh (thay đổi modules, stop, v.v.)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if msg.get("type") == "set_modules":
                    active_modules = parse_modules(msg.get("modules", ""))
                    if not await safe_send_json(websocket, {
                        "type": "log",
                        "msg": f"Modules cập nhật: {', '.join(active_modules) or 'none'}",
                        "level": "system",
                    }):
                        break
                elif msg.get("type") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except RuntimeError:
                break
            except Exception as exc:
                print(f"[WS] receive warning cam={cam_id}: {exc}")

            ret, frame = cap.read()
            if not ret:
                if is_local_file_source(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if not await safe_send_json(websocket, {
                    "type": "log",
                    "msg": "Mất tín hiệu hoặc không đọc được frame.",
                    "level": "error",
                }):
                    connected = False
                break

            # FIX #6 — chạy pipeline theo module flag
            frame, ai_logs = process_frame_with_modules(frame, active_modules)

            # FIX #4 — chỉ emit log thật từ pipeline (không fake)
            for log_msg in ai_logs:
                if not await safe_send_json(websocket, {"type": "log", "msg": log_msg, "level": "ai"}):
                    connected = False
                    break
            if not connected:
                break

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                await asyncio.sleep(0.03)
                continue

            jpeg_bytes = buffer.tobytes()

            # FIX #7 — lưu vào frame buffer (không ghi ra disk tự động)
            _frame_buffers[session_id].append(jpeg_bytes)

            frame_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            if not await safe_send_json(websocket, {"type": "image", "data": frame_b64}):
                break

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print(f"[System] Client ngắt kết nối {cam_id}")
    finally:
        try:
            cap.release()
        except Exception as exc:
            print(f"[System] cap.release warning cam={cam_id}: {exc}")
        print(f"[System] Stream closed cam={cam_id}, session={session_id}")
