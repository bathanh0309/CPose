from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.metrics import base_result, bbox_iou, detect_gt_available, gt_file_for, load_json, mean, metric_files, prediction_files


def evaluate_detection(pred_json: Path, gt_json: Path | None, iou_threshold: float = 0.5) -> dict[str, Any]:
    pred_records = load_json(pred_json, [])
    if gt_json is None or not gt_json.exists():
        total_pred = sum(len(frame.get("detections", [])) for frame in pred_records)
        return {"metric_type": "proxy", "total_pred_persons": total_pred, "warning": "ground truth missing"}
    gt_payload = load_json(gt_json, {})
    gt_by_frame = {int(frame.get("frame_id", -1)): frame.get("persons", []) for frame in gt_payload.get("frames", [])}
    tp = fp = fn = total_gt = total_pred = 0
    for record in pred_records:
        preds = record.get("detections", [])
        gt_persons = [p for p in gt_by_frame.get(int(record.get("frame_id", -1)), []) if p.get("visible", True)]
        total_gt += len(gt_persons)
        total_pred += len(preds)
        matched: set[int] = set()
        for pred in preds:
            best_iou = 0.0
            best_idx = None
            for idx, gt_person in enumerate(gt_persons):
                if idx in matched:
                    continue
                score = bbox_iou(pred.get("bbox", []), gt_person.get("bbox", []))
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_threshold:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gt_persons) - len(matched)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "metric_type": "ground_truth",
        "total_gt_persons": total_gt,
        "total_pred_persons": total_pred,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate(run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    has_gt = detect_gt_available(gt_dir, "detection")
    result = base_result("detection", has_gt)
    rows = [load_json(path, {}) for path in metric_files(run_dir, "detection_metrics.json")]
    gt_metrics = []
    if has_gt:
        for pred in prediction_files(run_dir / "1_detection", "detections.json"):
            gt_metrics.append(evaluate_detection(pred, gt_file_for(gt_dir, "detection", pred.parent.name)))
    tp = sum(int(row.get("true_positive", 0)) for row in gt_metrics)
    fp = sum(int(row.get("false_positive", 0)) for row in gt_metrics)
    fn = sum(int(row.get("false_negative", 0)) for row in gt_metrics)
    precision = tp / (tp + fp) if has_gt and tp + fp else None
    recall = tp / (tp + fn) if has_gt and tp + fn else None
    result.update({
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
        "mAP@50": None,
        "avg_detection_confidence": mean([float(row["avg_confidence"]) for row in rows if row.get("avg_confidence") is not None]),
        "avg_persons_per_frame": mean([float(row["avg_persons_per_frame"]) for row in rows if row.get("avg_persons_per_frame") is not None]),
        "total_detections": sum(int(row.get("total_person_detections", 0)) for row in rows),
        "evaluated_files": len(rows),
    })
    return result
