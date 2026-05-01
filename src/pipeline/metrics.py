from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def detect_gt_available(gt_dir: Path, module: str) -> bool:
    gt_module_dir = gt_dir / f"{module}_gt"
    return gt_module_dir.exists() and any(path.is_file() and path.name != ".gitkeep" for path in gt_module_dir.iterdir())


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def save_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return path


def metric_files(run_dir: Path, filename: str) -> list[Path]:
    return sorted(run_dir.rglob(filename))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def base_result(module: str, has_gt: bool) -> dict[str, Any]:
    return {
        "module": module,
        "metric_type": "ground_truth" if has_gt else "proxy",
        "failure_reason": "OK",
    }
