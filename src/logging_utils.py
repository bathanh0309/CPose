"""Terminal logging helpers for summary-only CPose modules."""
from __future__ import annotations

from pathlib import Path
from typing import Any


LINE = "=" * 60


def print_header(title: str) -> None:
    print(LINE)
    print(title)
    print(LINE)


def print_metric_table(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    width = max(len(str(key)) for key in metrics)
    for key, value in metrics.items():
        print(f"{key:<{width}} : {value}")


def print_video_progress(index: int, total: int, video_path: Path) -> None:
    print(f"\n[{index}/{total}] Processing {video_path.name}")


def print_saved(video_path: Path | None, json_path: Path | None, metric_path: Path | None = None) -> None:
    if video_path:
        print(f"Saved video : {video_path}")
    if json_path:
        print(f"Saved json  : {json_path}")
    if metric_path:
        print(f"Saved metric: {metric_path}")


def log_warning(message: str) -> None:
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


__all__ = [
    "LINE",
    "log_error",
    "log_warning",
    "print_header",
    "print_metric_table",
    "print_saved",
    "print_video_progress",
]
