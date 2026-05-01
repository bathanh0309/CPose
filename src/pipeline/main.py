from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import importlib
from pathlib import Path
from typing import Any

from src.evaluation.metrics import save_csv, save_json


def _flatten(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module, metrics in summary.get("modules", {}).items():
        row = {"module": module}
        for key, value in metrics.items():
            row[key] = value if not isinstance(value, (dict, list)) else str(value)
        rows.append(row)
    return rows


def evaluate_all(outputs: str | Path, gt: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    run_dir = Path(outputs)
    gt_dir = Path(gt)
    eval_out = Path(out) if out else run_dir / "evaluation"
    if not gt_dir.exists():
        print(f"[WARN] GT directory not found: {gt_dir}. Proxy metrics only.")
    modules = {}
    for name in ("detection", "tracking", "pose", "adl", "reid"):
        evaluator = importlib.import_module(f"src.evaluation.{name}_eval")
        modules[name] = evaluator.evaluate(run_dir, gt_dir)
    metric_type = "ground_truth" if any(item["metric_type"] == "ground_truth" for item in modules.values()) else "proxy"
    summary = {"metric_type": metric_type, "failure_reason": "OK", "run_dir": str(run_dir), "gt_dir": str(gt_dir), "modules": modules}
    save_json(eval_out / "evaluation_summary.json", summary)
    save_csv(eval_out / "evaluation_summary.csv", _flatten(summary))
    if modules["adl"]["metric_type"] == "ground_truth" and modules["adl"].get("confusion_matrix") is not None:
        save_json(eval_out / "confusion_matrix_adl.json", modules["adl"]["confusion_matrix"])
    if modules["reid"]["metric_type"] == "ground_truth":
        save_json(eval_out / "reid_error_cases.json", [])
    print(f"[INFO] Evaluation saved: {eval_out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CPose pipeline outputs")
    parser.add_argument("--outputs", required=True, help="Pipeline run directory")
    parser.add_argument("--gt", default="data/annotations", help="Annotation directory")
    parser.add_argument("--out", default=None, help="Evaluation output directory")
    args = parser.parse_args()
    evaluate_all(args.outputs, args.gt, args.out)


if __name__ == "__main__":
    main()
