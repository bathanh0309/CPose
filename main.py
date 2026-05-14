import asyncio
import base64
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "data" / "config" / "resources.txt"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
STATIC_DIR = BASE_DIR / "static"

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
                cameras.append({"name": name.strip(), "url": url.strip()})
    return cameras


def resolve_video_source(source: str | None, url: str | None) -> tuple[str | None, str]:
    if source and source.startswith("camera:"):
        try:
            camera_index = int(source.split(":", 1)[1])
        except ValueError:
            return None, "Camera khong hop le"

        cameras = read_camera_sources()
        if camera_index < 0 or camera_index >= len(cameras):
            return None, "Camera khong ton tai"
        return cameras[camera_index]["url"], cameras[camera_index]["name"]

    if source:
        return source, "File upload"

    if url:
        return url, "Nguon RTSP"

    return None, "Chua chon nguon video"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/style.css")
def stylesheet():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/scripts.js")
def scripts():
    return FileResponse(STATIC_DIR / "scripts.js", media_type="application/javascript")


@app.get("/api/cameras")
def get_cameras():
    cameras = [
        {"id": f"camera:{index}", "name": camera["name"], "display": camera["name"]}
        for index, camera in enumerate(read_camera_sources())
    ]

    return JSONResponse(content=cameras)


@app.post("/api/upload")
async def upload_video(file: Annotated[UploadFile, File(...)]):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "uploaded-video").name
    target = UPLOAD_DIR / safe_name

    suffix = target.suffix
    stem = target.stem
    counter = 1
    while target.exists():
        target = UPLOAD_DIR / f"{stem}-{counter}{suffix}"
        counter += 1

    with target.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    return {"name": target.name, "source": str(target)}


@app.websocket("/ws/stream/{cam_id}")
async def stream_video(
    websocket: WebSocket,
    cam_id: str,
    source: str | None = None,
    url: str | None = None,
):
    await websocket.accept()
    video_source, source_label = resolve_video_source(source, url)
    if not video_source:
        await websocket.send_json({"type": "log", "msg": source_label, "level": "error"})
        await websocket.close()
        return

    print(f"[System] Khoi tao luong {cam_id}: {source_label}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        await websocket.send_json(
            {"type": "log", "msg": f"Loi: Khong the mo nguon video {source_label}", "level": "error"}
        )
        await websocket.close()
        return

    await websocket.send_json(
        {"type": "log", "msg": f"Ket noi thanh cong toi {source_label}", "level": "system"}
    )

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_local_file_source(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                await websocket.send_json(
                    {"type": "log", "msg": "Mat tin hieu hoac khong doc duoc frame.", "level": "error"}
                )
                break

            # Chen pipeline CPose tai day:
            # frame, ai_logs = cpose_pipeline.process(frame)

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                await asyncio.sleep(0.03)
                continue

            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json({"type": "image", "data": frame_b64})

            frame_count += 1
            if frame_count % 90 == 0:
                await websocket.send_json(
                    {"type": "log", "msg": "Fast-ReID: Gan ID #001", "level": "ai"}
                )

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print(f"[System] Client da ngat ket noi {cam_id}")
    finally:
        cap.release()
