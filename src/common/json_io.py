from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path | None, default: Any = None) -> Any:
    if path is None:
        return default
    json_path = Path(path)
    if not json_path.exists():
        return default
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def save_csv(path: str | Path, rows: list[dict[str, Any]] | dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_list = [rows] if isinstance(rows, dict) else list(rows)
    if not row_list:
        output.write_text("", encoding="utf-8")
        return output
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        writer.writerows(row_list)
    return output


__all__ = ["load_json", "save_csv", "save_json"]
