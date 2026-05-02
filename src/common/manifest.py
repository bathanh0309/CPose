from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.common.video_io import (
    camera_id_from_video_name,
    datetime_sort_value,
    list_video_files,
    parse_video_start_time,
)


@dataclass(slots=True)
class VideoManifestItem:
    video: str
    camera_id: str
    start_time: str
    fps: float | None = None
    location: str | None = None
    timezone: str | None = None

    @property
    def stem(self) -> str:
        return Path(self.video).stem


@dataclass(slots=True)
class ResolvedVideoItem:
    path: Path
    video: str
    stem: str
    camera_id: str
    start_time: datetime | None
    fps: float | None
    location: str | None
    timezone: str | None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_multicam_manifest(path: str | Path | None) -> list[VideoManifestItem]:
    if path is None or not Path(path).exists():
        print("[WARN] Manifest not found. Falling back to filename timestamp parsing.")
        return []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[WARN] Invalid manifest {path}: {exc}. Falling back to filename timestamp parsing.")
        return []

    rows = payload if isinstance(payload, list) else payload.get("videos", []) if isinstance(payload, dict) else []
    items: list[VideoManifestItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        video = row.get("video") or row.get("path")
        if not video:
            continue
        video_path = Path(str(video))
        camera_id = str(row.get("camera_id") or video_path.stem.split("_")[0])
        start_time = str(row.get("start_time") or "")
        items.append(VideoManifestItem(
            video=str(video),
            camera_id=camera_id,
            start_time=start_time,
            fps=float(row["fps"]) if row.get("fps") is not None else None,
            location=row.get("location"),
            timezone=row.get("timezone"),
        ))
    return items


def parse_video_timestamp_from_filename(video_path: Path) -> tuple[str, datetime | None]:
    camera_id = camera_id_from_video_name(video_path)
    parsed = parse_video_start_time(video_path)
    return camera_id, parsed


def resolve_videos_from_manifest(input_dir: Path, manifest_path: Path | None) -> list[ResolvedVideoItem]:
    input_dir = Path(input_dir)
    manifest_items = load_multicam_manifest(manifest_path)
    resolved: list[ResolvedVideoItem] = []
    if manifest_items:
        for item in manifest_items:
            candidate = Path(item.video)
            path = candidate if candidate.is_absolute() else input_dir / candidate
            if not path.exists():
                print(f"[WARN] Manifest video not found: {path}")
            start_time = _parse_datetime(item.start_time) or parse_video_start_time(path)
            resolved.append(ResolvedVideoItem(
                path=path,
                video=item.video,
                stem=Path(item.video).stem,
                camera_id=item.camera_id,
                start_time=start_time,
                fps=item.fps,
                location=item.location,
                timezone=item.timezone,
            ))
    else:
        for video in list_video_files(input_dir):
            camera_id, start_time = parse_video_timestamp_from_filename(video)
            resolved.append(ResolvedVideoItem(
                path=video,
                video=video.name,
                stem=video.stem,
                camera_id=camera_id,
                start_time=start_time,
                fps=None,
                location=None,
                timezone=None,
            ))
    return sort_video_items(resolved)


def _resolved_item_sort_key(item: ResolvedVideoItem) -> tuple[int, float, str, str]:
    start_time = item.start_time or parse_video_start_time(item.path)
    return (
        0 if start_time is not None else 1,
        datetime_sort_value(start_time),
        item.camera_id.lower(),
        item.stem.lower(),
    )


def sort_video_items(items: list[ResolvedVideoItem]) -> list[ResolvedVideoItem]:
    return sorted(
        items,
        key=_resolved_item_sort_key,
    )


def get_item_by_stem(items: list[ResolvedVideoItem], stem: str) -> ResolvedVideoItem | None:
    for item in items:
        if item.stem == stem:
            return item
    return None
