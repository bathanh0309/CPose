"""Filesystem utilities for resources, outputs, and storage limits."""

from __future__ import annotations

import logging
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.utils.runtime_config import get_runtime_section

logger = logging.getLogger("[Storage]")

_STORAGE_CFG = get_runtime_section("storage")
PRUNE_TARGET_RATIO = float(_STORAGE_CFG.get("prune_target_ratio", 0.8))
MULTICAM_CAMERA_ORDER = ("cam01", "cam02", "cam03", "cam04")

_MULTICAM_SORT_FORMATS = (
    "%S-%M-%H-%d-%m-%Y",
    "%Y-%m-%d-%H-%M-%S",
)


def _parse_multicam_timestamp(item: Path | str) -> tuple[datetime, int, str] | None:
    """Parse a multicam clip stem into a sortable timestamp."""
    name = Path(item).stem
    parts = name.split("-")
    if len(parts) != 7:
        return None

    cam_token = parts[0]
    if not cam_token.startswith("cam"):
        return None

    cam_digits = "".join(ch for ch in cam_token if ch.isdigit())
    if not cam_digits:
        return None

    tail = "-".join(parts[1:])
    for fmt in _MULTICAM_SORT_FORMATS:
        try:
            clip_dt = datetime.strptime(tail, fmt)
            return clip_dt, int(cam_digits), name.lower()
        except ValueError:
            continue
    return None


def extract_multicam_camera_id(item: Path | str) -> str | None:
    """Return normalized camera id like cam01 when present."""
    name = Path(item).stem
    prefix = name.split("-", 1)[0].lower()
    if not prefix.startswith("cam"):
        return None

    cam_digits = "".join(ch for ch in prefix if ch.isdigit())
    if not cam_digits:
        return None
    return f"cam{int(cam_digits):02d}"


def multicam_sort_key(item: Path | str) -> tuple[datetime, int, str]:
    """Sort key that prefers timestamp, then camera index, then name."""
    parsed = _parse_multicam_timestamp(item)
    if parsed is not None:
        return parsed

    path = Path(item)
    try:
        fallback_dt = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        fallback_dt = datetime.min
    return fallback_dt, 9999, path.stem.lower()


def sort_multicam_clips(clips: Iterable[Path]) -> list[Path]:
    """
    Route-aware, camera-priority, nearest-next scheduler.
    Follows priority: cam01 -> cam02 -> cam03 -> cam04
    Starts strictly from cam01, then finds the next valid temporal clip
    to continue the route rather than exhausting cameras safely or globally sorting ignoring context.
    """
    clip_list = list(clips)
    if not clip_list:
        return []

    parsed_clips = []
    for c in clip_list:
        parsed = _parse_multicam_timestamp(c)
        if parsed is not None:
            dt, cam_idx, _ = parsed
            cam_str = f"cam{cam_idx:02d}"
            parsed_clips.append({"path": c, "dt": dt, "cam": cam_str})
        else:
            path = Path(c)
            try:
                fallback_dt = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                fallback_dt = datetime.min
            cam_str = extract_multicam_camera_id(c) or "unknown"
            parsed_clips.append({"path": c, "dt": fallback_dt, "cam": cam_str})

    available = parsed_clips.copy()
    result = []
    
    cam_order_map = {cam: i for i, cam in enumerate(MULTICAM_CAMERA_ORDER)}

    current_time = None
    current_cam_idx = -1

    while available:
        selected_clip = None

        if current_time is None:
            # Rule: Start from cam01 first
            cam01_clips = [c for c in available if c["cam"] == "cam01"]
            if cam01_clips:
                selected_clip = min(cam01_clips, key=lambda x: x["dt"])
            else:
                selected_clip = min(available, key=lambda x: x["dt"])
        else:
            # Rule: choose the clip whose timestamp is closest to current route context
            # We filter clips that are >= current_time to simulate forward progression.
            future_clips = [c for c in available if c["dt"] >= current_time]

            if not future_clips:
                # If everything remaining is in the past, reset the route context to process them.
                current_time = None
                current_cam_idx = -1
                continue

            # Pick the nearest available clip in the future.
            # If tie, camera traversal priority (cam01, then cam02...) applies.
            selected_clip = min(future_clips, key=lambda x: (x["dt"], cam_order_map.get(x["cam"], 999)))

        result.append(selected_clip["path"])
        available.remove(selected_clip)
        current_time = selected_clip["dt"]
        current_cam_idx = cam_order_map.get(selected_clip["cam"], 999)

    return result


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
