"""Filesystem utilities for resources, outputs, and storage limits."""

from __future__ import annotations

import logging
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Iterable
import re

from app.utils.runtime_config import get_runtime_section

logger = logging.getLogger("[Storage]")

_STORAGE_CFG = get_runtime_section("storage")
PRUNE_TARGET_RATIO = float(_STORAGE_CFG.get("prune_target_ratio", 0.8))
MULTICAM_CAMERA_ORDER = ("cam01", "cam02", "cam03", "cam04")

_MULTICAM_TIMESTAMP_PATTERNS = (
    (re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?P<sep>[_-])(?P<time>\d{2}-\d{2}-\d{2})"), "%Y-%m-%d_%H-%M-%S"),
    (re.compile(r"(?P<date>\d{2}-\d{2}-\d{4})(?P<sep>[_-])(?P<time>\d{2}-\d{2}-\d{2})"), "%d-%m-%Y_%H-%M-%S"),
    (re.compile(r"(?P<date>\d{8})(?P<sep>[_-])(?P<time>\d{6})"), "%Y%m%d_%H%M%S"),
)


def _parse_multicam_timestamp(item: Path | str) -> tuple[datetime, int, str] | None:
    """Parse a multicam clip stem into a sortable timestamp."""
    name = Path(item).stem
    for pattern, fmt in _MULTICAM_TIMESTAMP_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        sep = match.group("sep")
        text = f"{match.group('date')}{sep}{match.group('time')}"
        try:
            clip_dt = datetime.strptime(text, fmt if sep == "_" else fmt.replace("_", "-"))
            return clip_dt, 0, name.lower()
        except ValueError:
            continue
    return None


def extract_multicam_camera_id(item: Path | str) -> str | None:
    """Return normalized camera id like cam01 when present."""
    name = Path(item).stem
    match = re.search(r"cam0*(\d+)", name, flags=re.IGNORECASE)
    if not match:
        return None

    cam_digits = match.group(1)
    if not cam_digits:
        return None
    return f"cam{int(cam_digits):02d}"


def multicam_sort_key(item: Path | str) -> tuple[datetime, int, str]:
    """Sort key that prefers timestamp, then file name for deterministic order."""
    parsed = _parse_multicam_timestamp(item)
    if parsed is not None:
        return parsed

    path = Path(item)
    try:
        fallback_dt = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        fallback_dt = datetime.min
    return fallback_dt, 0, path.stem.lower()


def sort_multicam_clips(clips: Iterable[Path]) -> list[Path]:
    """Sort clips strictly by timestamp in the clip name, then file name."""
    clip_list = list(clips)
    return sorted(clip_list, key=multicam_sort_key)


class StorageManager:
    """Project filesystem helper."""

    def parse_resources(self, resources_file: Path) -> list[dict]:
        cameras: list[dict] = []
        if not resources_file.exists():
            return cameras

        camera_idx = 1
        for raw_line in resources_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Support "Name __ URL" format
            if "__" in line:
                label, url = [p.strip() for p in line.split("__", 1)]
            else:
                label = f"Camera {camera_idx}"
                url = line

            cameras.append({
                "cam_id": f"{camera_idx:02d}",
                "label": label,
                "rtsp_url": url
            })
            camera_idx += 1

        logger.info("Parsed %d cameras from %s", len(cameras), resources_file.name)
        return cameras

    def list_videos(self, raw_videos_dir: Path) -> list[dict]:
        if not raw_videos_dir.exists():
            return []

        files = sorted(raw_videos_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
        return [
            {
                "filename": item.name,
                "size_mb": round(item.stat().st_size / 1e6, 2),
                "mtime": item.stat().st_mtime,
            }
            for item in files
        ]

    def list_results(self, output_dir: Path) -> list[dict]:
        if not output_dir.exists():
            return []

        results: list[dict] = []
        for folder in sort_multicam_clips(path for path in output_dir.iterdir() if path.is_dir()):
            if not folder.is_dir():
                continue
            png_files = sorted(folder.glob("*.png"))
            label_file = next(iter(sorted(folder.glob("*_labels.txt"))), None)
            results.append(
                {
                    "clip_stem": folder.name,
                    "frames": len(png_files),
                    "label_count": self._count_data_lines(label_file),
                    "label_file": label_file.name if label_file else None,
                    "preview_frame": png_files[0].name if png_files else None,
                }
            )
        return results

    def list_pose_results(self, output_dir: Path) -> list[dict]:
        if not output_dir.exists():
            return []

        results: list[dict] = []
        from app import BASE_DIR
        out_process_dir = BASE_DIR / "data" / "output_process"

        for folder in sort_multicam_clips(path for path in output_dir.iterdir() if path.is_dir()):
            if not folder.is_dir():
                continue

            keypoints_file = next(iter(sorted(folder.glob("*_keypoints.txt"))), None)
            adl_file = next(iter(sorted(folder.glob("*_adl.txt"))), None)
            overlay_files = list(folder.glob("*_overlay_*.png"))
            adl_summary = self._read_adl_summary(adl_file) if adl_file else {}
            
            mp4_path = out_process_dir / folder.name / f"{folder.name}_processed.mp4"
            preview_path = folder / f"{folder.name}_preview.mp4"

            results.append(
                {
                    "clip_stem": folder.name,
                    "keypoints_count": self._count_data_lines(keypoints_file),
                    "adl_events": self._count_data_lines(adl_file),
                    "overlays": len(overlay_files),
                    "adl_summary": adl_summary,
                    "mp4_exists": mp4_path.exists(),
                    "preview_exists": preview_path.exists(),
                }
            )
        return results

    def get_pose_summary(self, output_dir: Path, clip_stem: str) -> dict | None:
        clip_dir = output_dir / clip_stem
        if not clip_dir.is_dir():
            return None
        adl_file = next(iter(sorted(clip_dir.glob("*_adl.txt"))), None)
        if adl_file is None:
            return {}
        return self._read_adl_summary(adl_file)

    def storage_info(self, raw_videos_dir: Path) -> dict:
        files = list(raw_videos_dir.glob("*.mp4")) if raw_videos_dir.exists() else []
        total_bytes = sum(item.stat().st_size for item in files)
        return {
            "used_gb": round(total_bytes / 1e9, 3),
            "used_mb": round(total_bytes / 1e6, 1),
            "file_count": len(files),
        }

    def enforce_storage_limit(self, raw_videos_dir: Path, limit_gb: float) -> None:
        limit_bytes = limit_gb * 1e9
        target_bytes = limit_bytes * PRUNE_TARGET_RATIO
        files = sorted(raw_videos_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime)
        total_bytes = sum(item.stat().st_size for item in files)

        if total_bytes <= limit_bytes:
            return

        logger.warning(
            "Storage %.2f GB exceeds limit %.2f GB, pruning oldest clips",
            total_bytes / 1e9,
            limit_gb,
        )

        for item in files:
            if total_bytes <= target_bytes:
                break
            size = item.stat().st_size
            item.unlink(missing_ok=True)
            total_bytes -= size
            logger.info("Deleted oldest clip: %s (%.1f MB)", item.name, size / 1e6)

        from app import socketio

        socketio.emit(
            "storage_warning",
            {
                "used_gb": round(total_bytes / 1e9, 3),
                "limit_gb": limit_gb,
                "pct": round((total_bytes / limit_bytes) * 100) if limit_bytes else 0,
            },
        )

    @staticmethod
    def _count_data_lines(file_path: Path | None) -> int:
        if file_path is None or not file_path.exists():
            return 0
        count = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    count += 1
        return count

    @staticmethod
    def _read_adl_summary(adl_file: Path) -> dict[str, float]:
        labels: Counter[str] = Counter()
        with adl_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 4:
                    labels[parts[2]] += 1

        total = sum(labels.values())
        if total == 0:
            return {}
        return {label: round((count / total) * 100, 1) for label, count in sorted(labels.items())}


FileHandler = StorageManager
