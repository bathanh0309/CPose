from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.metrics import base_result, detect_gt_available, load_json, metric_files


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "adl")
    result = base_result("adl", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "adl_metrics.json")]
    distribution: Counter[str] = Counter()
    for row in rows:
        distribution.update(row.get("class_distribution", {}))
    total = sum(distribution.values())
    result.update({
        "accuracy": None,
        "macro_f1": None,
        "per_class_precision": None,
        "per_class_recall": None,
        "confusion_matrix": None,
        "unknown_rate": distribution.get("unknown", 0) / total if total else None,
        "class_distribution": dict(distribution),
        "evaluated_files": len(rows),
    })
    return result
