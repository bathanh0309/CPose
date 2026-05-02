from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from pathlib import Path
from typing import Any

from src import ANNOTATIONS_DIR, OUTPUT_DIR, print_module_console
from src.evaluation import adl_eval, detection_eval, pose_eval, reid_eval, tracking_eval
from src.common.json_io import save_csv, save_json


def evaluate_all(run_dir: str | Path, gt_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    gt_path = Path(gt_dir)
    out_path = Path(output_dir)
    modules = {
        "detection": detection_eval.evaluate(run_path, gt_path),
        "tracking": tracking_eval.evaluate(run_path, gt_path),
        "pose": pose_eval.evaluate(run_path, gt_path),
        "adl": adl_eval.evaluate(run_path, gt_path),
        "reid": reid_eval.evaluate(run_path, gt_path),
    }
    metric_type = "ground_truth" if any(module.get("metric_type") == "ground_truth" for module in modules.values()) else "proxy"
    summary = {
        "metric_type": metric_type,
        "modules": modules,
        "failure_reason": "OK" if metric_type == "ground_truth" else "EVALUATION_SKIPPED_NO_GT",
    }
    save_json(out_path / "evaluation_summary.json", summary)
    rows = [{"module": name, **payload} for name, payload in modules.items()]
    save_csv(out_path / "evaluation_summary.csv", rows)
    save_json(out_path / "confusion_matrix_adl.json", modules["adl"].get("confusion_matrix") or {})
    save_json(out_path / "reid_error_cases.json", [])
    print(f"[INFO] Evaluation summary saved: {out_path / 'evaluation_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CPose pipeline outputs")
    parser.add_argument("--outputs", default=str(OUTPUT_DIR))
    parser.add_argument("--gt", default=str(ANNOTATIONS_DIR), help="Train/val annotation root")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "evaluation"))
    args = parser.parse_args()
    print_module_console("evaluation", args)
    evaluate_all(args.outputs, args.gt, args.out)


if __name__ == "__main__":
    main()
