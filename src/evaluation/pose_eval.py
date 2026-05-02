from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.detection_eval import base_result, detect_gt_available, load_json, mean, metric_files


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "pose")
    result = base_result("pose", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "pose_metrics.json")]
    result.update({
        "PCK@0.1": None,
        "mean_keypoint_confidence": mean([float(row["mean_keypoint_confidence"]) for row in rows if row.get("mean_keypoint_confidence") is not None]),
        "visible_keypoint_ratio": mean([float(row["visible_keypoint_ratio"]) for row in rows if row.get("visible_keypoint_ratio") is not None]),
        "missing_keypoint_rate": mean([float(row["missing_keypoint_rate"]) for row in rows if row.get("missing_keypoint_rate") is not None]),
        "total_pose_instances": sum(int(row.get("total_pose_instances", 0)) for row in rows),
        "evaluated_files": len(rows),
    })
    return result
