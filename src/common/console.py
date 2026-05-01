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

