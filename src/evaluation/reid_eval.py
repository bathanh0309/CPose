from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.metrics import base_result, detect_gt_available, load_json, metric_files


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "global_id")
    result = base_result("reid", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "reid_metrics.json")]
    result.update({
        "global_id_accuracy": None,
        "cross_camera_idf1": None,
        "false_split_rate": None,
        "false_merge_rate": None,
        "transfer_success_rate": None,
        "blind_zone_recovery_rate": None,
        "clothing_change_preservation": None,
        "global_id_count": sum(int(row.get("unique_global_ids", 0)) for row in rows) if rows else None,
        "unknown_rate": None,
        "evaluated_files": len(rows),
    })
    return result
