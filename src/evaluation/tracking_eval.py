from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.metrics import base_result, detect_gt_available, load_json, mean, metric_files


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "tracking")
    result = base_result("tracking", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "tracking_metrics.json")]
    result.update({
        "idf1": None,
        "id_switch_count": None,
        "fragmentation_count": None,
        "track_purity": None,
        "proxy_track_fragmentation": sum(int(row.get("proxy_track_fragmentation", row.get("track_fragmentation_proxy", 0))) for row in rows),
        "active_track_count_mean": mean([float(row["active_track_count_mean"]) for row in rows if row.get("active_track_count_mean") is not None]),
        "total_tracks": sum(int(row.get("total_tracks", 0)) for row in rows),
        "evaluated_files": len(rows),
    })
    return result
