from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from src.common.paths import resolve_path
from src.common.schemas import to_dict


class Timer:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def fps(processed_frames: int, elapsed_sec: float) -> float:
    return processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0


def latency_ms(processed_frames: int, elapsed_sec: float) -> float:
    return (elapsed_sec / processed_frames * 1000.0) if processed_frames > 0 else 0.0


def save_json(path: str | Path, payload: Any) -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_dict(payload), handle, indent=2, ensure_ascii=False)
    return output_path


def load_json(path: str | Path, default: Any = None) -> Any:
    input_path = resolve_path(path)
    if not input_path.exists():
        return default
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return output_path


def summarize_metrics(metric_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_file in metric_files:
        payload = load_json(metric_file, {})
        if isinstance(payload, dict):
            rows.append({"metric_file": str(metric_file), **payload})
    return rows

