from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_error_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _warn_optional(name: str, value: str | None) -> str | None:
    if not value:
        _log("WARN", f"Optional --{name} not provided; continuing with defaults/proxy behavior.")
        return None
    path = Path(value)
    if not path.exists():
        _log("WARN", f"Optional --{name} path not found: {path}; continuing with defaults/proxy behavior.")
    return value


def _call(module_name: str, function_name: str, *args: Any, **kwargs: Any) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(*args, **kwargs)


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest: str | None = None,
    topology: str | None = None,
    config: str | None = None,
    gt: str | None = None,
) -> Path:
    input_path = Path(input_dir)
    output_base = Path(output_dir)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / "pipeline" / run_ts
    detection_dir = run_dir / "1_detection"
    tracking_dir = run_dir / "2_tracking"
    pose_dir = run_dir / "3_pose"
    adl_dir = run_dir / "4_adl"
    reid_dir = run_dir / "5_reid"
    manifest = _warn_optional("manifest", manifest)
    topology = _warn_optional("topology", topology)
    config = _warn_optional("config", config)
    gt = _warn_optional("gt", gt)
    effective_config = {
        "run_timestamp": run_ts,
        "paths": {
            "input": input_path,
            "run_dir": run_dir,
            "detection": detection_dir,
            "tracking": tracking_dir,
            "pose": pose_dir,
            "adl": adl_dir,
            "reid": reid_dir,
            "manifest": manifest,
            "topology": topology,
            "config": config,
            "gt": gt,
        },
        "fallbacks": {
            "manifest_missing": manifest is None or not Path(manifest).exists(),
            "topology_missing": topology is None or not Path(topology).exists(),
            "config_missing": config is None or not Path(config).exists(),
            "gt_missing": gt is None or not Path(gt).exists(),
        },
    }
    _write_run_config(run_dir / "run_config.yaml", effective_config)
    _log("INFO", f"Pipeline run directory: {run_dir}")

    steps = [
        ("Detection", "src.human_detection.api", "process_folder", (input_path, detection_dir), {}),
        ("Tracking", "src.human_tracking.api", "process_folder", (input_path, tracking_dir), {"detection_dir": detection_dir}),
        ("Pose", "src.pose_estimation.api", "process_folder", (input_path, pose_dir), {"track_dir": tracking_dir}),
        ("ADL", "src.adl_recognition.api", "process_folder", (pose_dir, input_path, adl_dir), {}),
        ("ReID", "src.global_reid.api", "process_folder", (input_path, reid_dir), {"pose_dir": pose_dir, "adl_dir": adl_dir}),
    ]
    for label, module_name, function_name, args, kwargs in steps:
        _log("INFO", f"Step: {label}")
        try:
            _call(module_name, function_name, *args, **kwargs)
        except Exception as exc:
            _log("ERROR", f"{label} step failed: {exc}")
            error_dir = run_dir / label.lower()
            _write_error_json(error_dir / "error.json", {"failure_reason": f"{label.upper()}_STEP_FAILED", "error": str(exc)})

    if gt:
        _log("INFO", "Step: Evaluation")
        try:
            _call("src.evaluation.main", "evaluate_all", run_dir, Path(gt), run_dir / "evaluation")
        except Exception as exc:
            _log("ERROR", f"Evaluation failed: {exc}")

    _log("INFO", "Step: Benchmark")
    try:
        _call("src.pipeline.benchmark_all", "benchmark", run_dir)
    except Exception as exc:
        _log("ERROR", f"Benchmark failed: {exc}")
    _log("INFO", f"Pipeline complete: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPose TFCS-PAR terminal pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--gt", default=None)
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.manifest, args.topology, args.config, args.gt)


if __name__ == "__main__":
    main()
