import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Cắm đường ống nối sang file làm việc thực sự
from app.api.routes_process import router as phase2_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="MultiCam Surveillance API",
    description="Phase 1: RTSP Collection | Phase 2: Auto-Labeling",
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Link hai file lại với nhau (Rút gọn toàn bộ logic) ───────────────────
app.include_router(phase2_router)

# ─── Phục vụ file giao diện người dùng Web ────────────────────────────
_static_dir = Path("static")
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
