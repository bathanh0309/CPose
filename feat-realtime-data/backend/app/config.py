# Cấu hình trung tâm của backend, đọc biến môi trường bằng Pydantic Settings.
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_DIR = _REPO_ROOT / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(_CONFIGS_DIR / "runtime.env"),
            str(_REPO_ROOT / ".env"),
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Runtime ────────────────────────────────────────────────────────────
    app_env: Literal["development", "production", "test"] = "development"
    api_port: int = 8000
    frontend_port: int = 3000
    frontend_origin: str = "http://localhost:3000"

    # ── Paths ──────────────────────────────────────────────────────────────
    recordings_dir: Path = _REPO_ROOT / "data" / "raw_videos"
    processed_videos_dir: Path = _REPO_ROOT / "data" / "processed_videos"

    # ── Preview ────────────────────────────────────────────────────────────
    preview_mode: Literal["mjpeg"] = "mjpeg"   # WebRTC upgrade path kept open
    preview_fps: float = 5.0                    # target FPS for MJPEG output
    preview_width: int = 640
    preview_height: int = 360

    # ── Detection ─────────────────────────────────────────────────────────
    detector_model: str = "yolo_nano"
    detector_device: Literal["cpu", "cuda", "mps"] = "cpu"
    detection_conf: float = Field(default=0.35, ge=0.0, le=1.0)
    detection_width: int = 320    # resize frame before inference to save CPU
    detection_height: int = 180

    # ── Recording ─────────────────────────────────────────────────────────
    clip_seconds_before: int = 2    # pre-roll buffer length in seconds
    clip_seconds_after: int = 8     # post-roll after last detection
    clip_max_seconds: int = 10      # split long detections into ~10s clips
    max_cameras: int = 4

    # ── Database ──────────────────────────────────────────────────────────
    db_path: Path = _REPO_ROOT / "data" / "rtsp_monitor.db"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
