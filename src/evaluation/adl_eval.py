from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.detection_eval import base_result, detect_gt_available, load_json, metric_files


def _f1(tp: int, fp: int, fn: int) -> float | None:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "adl")
    result = base_result("adl", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "adl_metrics.json")]
    distribution: Counter[str] = Counter()
    for row in rows:
        distribution.update(row.get("class_distribution", {}))
    total = sum(distribution.values())
    payload = {
        "unknown_rate": distribution.get("unknown", 0) / total if total else None,
        "class_distribution": dict(distribution),
        "evaluated_files": len(rows),
    }
    if has_gt:
        # Event-level GT support is intentionally conservative: match by track_id and frame range.
        pred_events = []
        for path in run_dir.rglob("adl_events.json"):
            pred_events.extend(load_json(path, []))
        gt_events = []
        for path in (gt_dir / "adl_gt").glob("*.json"):
            gt_events.extend(load_json(path, {}).get("events", []))
        y_true: list[str] = []
        y_pred: list[str] = []
        for pred in pred_events:
            frame_id = int(pred.get("frame_id", -1))
            track_id = int(pred.get("track_id", -999999))
            match = next((gt for gt in gt_events if int(gt.get("track_id", -1)) == track_id and int(gt.get("start_frame", -1)) <= frame_id <= int(gt.get("end_frame", -1))), None)
            if match:
                y_true.append(str(match.get("label")))
                y_pred.append(str(pred.get("adl_label", "unknown")))
        classes = sorted(set(y_true) | set(y_pred))
        confusion = {actual: {pred: 0 for pred in classes} for actual in classes}
        for actual, pred in zip(y_true, y_pred):
            confusion[actual][pred] += 1
        correct = sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred)
        per_precision = {}
        per_recall = {}
        f1_values = []
        for cls in classes:
            tp = confusion.get(cls, {}).get(cls, 0)
            fp = sum(confusion.get(other, {}).get(cls, 0) for other in classes if other != cls)
            fn = sum(value for pred, value in confusion.get(cls, {}).items() if pred != cls)
            per_precision[cls] = tp / (tp + fp) if tp + fp else None
            per_recall[cls] = tp / (tp + fn) if tp + fn else None
            value = _f1(tp, fp, fn)
            if value is not None:
                f1_values.append(value)
        payload.update({
            "accuracy": correct / len(y_true) if y_true else None,
            "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
            "per_class_precision": per_precision,
            "per_class_recall": per_recall,
            "confusion_matrix": confusion,
        })
    result.update(payload)
    return result
