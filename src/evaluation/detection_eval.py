from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.metrics import base_result, detect_gt_available, load_json, mean, metric_files


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "detection")
    result = base_result("detection", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "detection_metrics.json")]
    result.update({
        "precision": None,
        "recall": None,
        "f1": None,
        "mAP@50": None,
        "avg_detection_confidence": mean([float(row["avg_confidence"]) for row in rows if row.get("avg_confidence") is not None]),
        "avg_persons_per_frame": mean([float(row["avg_persons_per_frame"]) for row in rows if row.get("avg_persons_per_frame") is not None]),
        "total_detections": sum(int(row.get("total_person_detections", 0)) for row in rows),
        "evaluated_files": len(rows),
    })
    return result
