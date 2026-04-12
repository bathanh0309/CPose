# Parser đọc resources.txt, tạo cấu hình camera và che RTSP nhạy cảm.
"""
Loads camera configurations from resources.txt.

Supported line formats:
  1) Cam_Name__rtsp://user:pass@host/stream
  2) CamName=rtsp://user:pass@host/stream
  3) rtsp://user:pass@host/stream
  4) # comment lines and blank lines
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: str


def _parse_camera_line(line: str, index: int) -> tuple[str, str] | None:
    if "__" in line:
        name_part, *url_parts = line.split("__")
        name = name_part.strip()
        url = "__".join(url_parts).strip()
        if name and url:
            return name, url
        return None

    if "=" in line and not line.startswith("rtsp://"):
        name_part, _, url_part = line.partition("=")
        name = name_part.strip()
        url = url_part.strip()
        if name and url:
            return name, url
        return None

    if line.lower().startswith("rtsp://"):
        return f"Cam {index:02d}", line

    return None


def parse_cameras_text(text: str, max_cameras: int = 4) -> tuple[list[CameraConfig], int]:
    """Parse resources.txt content and return valid cameras plus skipped line count."""
    cameras: list[CameraConfig] = []
    skipped_lines = 0
    index = 1

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff")

        if not line or line.startswith("#"):
            continue

        if len(cameras) >= max_cameras:
            break

        parsed = _parse_camera_line(line, index)
        if not parsed:
            skipped_lines += 1
            continue

        name, url = parsed
        cam_id = f"cam{index:02d}"
        cameras.append(CameraConfig(id=cam_id, name=name, rtsp_url=url))
        index += 1

    return cameras, skipped_lines


def mask_rtsp_source(rtsp_url: str) -> str:
    """Hide credentials and sensitive host segments before showing RTSP on the UI."""
    if not rtsp_url:
        return ""

    try:
        parsed = urlsplit(rtsp_url)
    except ValueError:
        return "rtsp://***"

    scheme = parsed.scheme or "rtsp"
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""

    if host.count(".") == 3:
        octets = host.split(".")
        masked_host = ".".join([octets[0], octets[1], "xxx", "xxx"])
    elif host:
        masked_host = f"{host[:2]}***"
    else:
        masked_host = "***"

    auth = "***:***@" if parsed.username or parsed.password else ""
    path = parsed.path or ""
    if path and path != "/":
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) > 1:
            path = f"/.../{segments[-1]}"

    return f"{scheme}://{auth}{masked_host}{port}{path}"


def load_cameras(resources_path: Path, max_cameras: int = 4) -> list[CameraConfig]:
    """
    Parse resources.txt and return up to max_cameras CameraConfig entries.
    Gracefully handles missing file by returning an empty list.
    """
    if not resources_path.exists():
        return []

    cameras, _ = parse_cameras_text(
        resources_path.read_text(encoding="utf-8"),
        max_cameras=max_cameras,
    )
    return cameras
