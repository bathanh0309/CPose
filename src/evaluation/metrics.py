from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


GT_DIR_MAP = {
    "detection": "detection_gt",
    "tracking": "tracking_gt",
    "pose": "pose_gt",
    "adl": "adl_gt",
    "global_id": "global_id_gt",
}


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_files(run_dir: Path, filename: str) -> list[Path]:
    return sorted(run_dir.rglob(filename))


def prediction_files(run_dir: Path, filename: str) -> list[Path]:
    return sorted(run_dir.rglob(filename))


def detect_gt_available(gt_dir: Path, module: str) -> bool:
    root = gt_dir / GT_DIR_MAP.get(module, module)
    return root.exists() and any(root.glob("*.json"))


def gt_file_for(gt_dir: Path, module: str, stem: str) -> Path | None:
    root = gt_dir / GT_DIR_MAP.get(module, module)
    candidate = root / f"{stem}.json"
    return candidate if candidate.exists() else None


def base_result(module: str, has_gt: bool) -> dict[str, Any]:
    return {
        "module": module,
        "metric_type": "ground_truth" if has_gt else "proxy",
        "failure_reason": "OK" if has_gt else "EVALUATION_SKIPPED_NO_GT",
    }


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0
