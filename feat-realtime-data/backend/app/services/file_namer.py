# Sinh tên file clip theo ngày và camera, tránh trùng khi ghi hình.
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _camera_label(camera_id: str) -> str:
    match = re.search(r"(\d+)$", camera_id)
    if match:
        return f"Cam{int(match.group(1)):02d}"
    return "Cam00"


def clip_path(recordings_dir: Path, camera_id: str, ts: datetime) -> Path:
    date_str = ts.strftime("%Y%m%d")
    recordings_dir.mkdir(parents=True, exist_ok=True)

    camera_label = _camera_label(camera_id)
    base_path = recordings_dir / f"{date_str}_{camera_label}.mp4"
    if not base_path.exists():
        return base_path

    index = 2
    while True:
        candidate = recordings_dir / f"{date_str}_{camera_label}_{index:02d}.mp4"
        if not candidate.exists():
            return candidate
        index += 1


def processed_clip_path(raw_clip_path: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    suffix = raw_clip_path.suffix or ".mp4"
    return processed_dir / f"{raw_clip_path.stem}_pose_adl{suffix}"
