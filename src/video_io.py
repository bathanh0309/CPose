from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.paths import resolve_path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
_FILENAME_TIMESTAMP_RE = re.compile(
    r"(?P<camera>cam\d+|[^_\-]+)[_\-](?P<date>\d{4}[_\-]\d{2}[_\-]\d{2})[_\-](?P<time>\d{2}[_\-]\d{2}[_\-]\d{2})",
    re.IGNORECASE,
)


@dataclass(slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def datetime_sort_value(value: datetime | None) -> float:
    if value is None:
        return float("inf")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).timestamp()
    return value.astimezone(timezone.utc).timestamp()


def camera_id_from_video_name(video_path: str | Path) -> str:
    path = Path(video_path)
    match = _FILENAME_TIMESTAMP_RE.search(path.stem)
    if match:
        return match.group("camera")
    return path.stem.split("_")[0].split("-")[0] or path.stem


def parse_video_start_time(video_path: str | Path) -> datetime | None:
    match = _FILENAME_TIMESTAMP_RE.search(Path(video_path).stem)
    if not match:
        return None
    raw = f"{match.group('date')}_{match.group('time')}".replace("-", "_")
    try:
        return datetime.strptime(raw, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return None


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return float("inf")


def video_sort_key(video_path: str | Path) -> tuple[int, float, str, str]:
    path = Path(video_path)
    start_time = parse_video_start_time(path)
    if start_time is not None:
        return (
            0,
            datetime_sort_value(start_time),
            camera_id_from_video_name(path).lower(),
            path.name.lower(),
        )
    resolved = resolve_path(path)
    return (
        1,
        _file_mtime(resolved),
        camera_id_from_video_name(path).lower(),
        path.name.lower(),
    )


def list_video_files(input_dir: str | Path) -> list[Path]:
    directory = resolve_path(input_dir)
    if not directory.exists():
        return []
    videos = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if any(parse_video_start_time(path) is None for path in videos):
        print("[WARN] Some video filenames have no parseable timestamp; using modified time fallback after timestamped videos.")
    return sorted(videos, key=video_sort_key)


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(resolve_path(video_path)))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return capture


def get_video_info(video_path: str | Path) -> VideoInfo:
    path = resolve_path(video_path)
    capture = open_video(path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(path, width, height, fps, frame_count, duration_sec)
    finally:
        capture.release()


def get_video_meta(video_path: str | Path) -> VideoInfo:
    return get_video_info(video_path)


def read_video_frames_to_memory(video_path: str | Path) -> list[np.ndarray]:
    return [frame for frame, _timestamp_sec, _frame_id in iter_video_frames(video_path)]


def read_video_frames(video_path: str | Path) -> list[np.ndarray]:
    print("[WARN] read_video_frames loads full video into memory. Prefer iter_video_frames().")
    return read_video_frames_to_memory(video_path)


def create_video_writer(output_path: str | Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps if fps > 0 else 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")
    return writer


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[np.ndarray, float, int]]:
    capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame, frame_id / fps if fps > 0 else 0.0, frame_id
            frame_id += 1
    finally:
        capture.release()


_PREVIEW_AVAILABLE: bool | None = None
_PREVIEW_DISABLED_WINDOWS: set[str] = set()
_PREVIEW_LAST_SHOW: dict[str, float] = {}


def show_frame_preview(window_name: str, frame: np.ndarray, fps: float | None = None) -> bool:
    """Show a preview frame at source-video speed.

    Returns True when the user presses Q/Esc or closes the preview window.
    Callers should then stop previewing while continuing to save output video.
    """
    global _PREVIEW_AVAILABLE
    if _PREVIEW_AVAILABLE is False or window_name in _PREVIEW_DISABLED_WINDOWS:
        return True
    try:
        target_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        last_show = _PREVIEW_LAST_SHOW.get(window_name)
        delay_ms = 1
        if last_show is not None:
            remaining = target_interval - (time.perf_counter() - last_show)
            if remaining > 0:
                delay_ms = max(1, int(round(remaining * 1000)))

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        _PREVIEW_LAST_SHOW[window_name] = time.perf_counter()
        if _PREVIEW_AVAILABLE is None:
            _PREVIEW_AVAILABLE = True
        if key in (ord("q"), ord("Q"), 27):
            _PREVIEW_AVAILABLE = False
            _PREVIEW_DISABLED_WINDOWS.add(window_name)
            _PREVIEW_LAST_SHOW.pop(window_name, None)
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
            return True
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                _PREVIEW_AVAILABLE = False
                _PREVIEW_DISABLED_WINDOWS.add(window_name)
                _PREVIEW_LAST_SHOW.pop(window_name, None)
                return True
        except cv2.error:
            _PREVIEW_AVAILABLE = False
            _PREVIEW_DISABLED_WINDOWS.add(window_name)
            _PREVIEW_LAST_SHOW.pop(window_name, None)
            return True
        return False
    except cv2.error:
        if _PREVIEW_AVAILABLE is None:
            print("[WARN] No display available - live preview disabled. MP4 will still be saved.")
            _PREVIEW_AVAILABLE = False
        return False


def reset_preview_state() -> None:
    """Call once per module run to re-evaluate display availability."""
    global _PREVIEW_AVAILABLE
    _PREVIEW_AVAILABLE = None
    _PREVIEW_DISABLED_WINDOWS.clear()
    _PREVIEW_LAST_SHOW.clear()
