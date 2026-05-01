from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.common.paths import resolve_path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


@dataclass(slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def list_video_files(input_dir: str | Path) -> list[Path]:
    directory = resolve_path(input_dir)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


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
