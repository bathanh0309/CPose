"""UTF-8 JSON and CSV helpers for CPose outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.common.paths import resolve_path
from src.common.schemas import to_dict


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


def write_json(path: str | Path, payload: Any) -> Path:
    return save_json(path, payload)


def read_json(path: str | Path, default: Any = None) -> Any:
    return load_json(path, default)


__all__ = ["load_json", "read_json", "save_csv", "save_json", "write_json"]
