from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import ANNOTATIONS_DIR, DATA_TEST_DIR, DEFAULT_CONFIG, OUTPUT_DIR, print_module_console
from src.common.errors import ErrorCode
from src.common.manifest import resolve_videos_from_manifest
from src.common.metrics import save_json
from src.common.model_registry import build_effective_runtime_config, get_section, load_model_registry, resolve_model_path
from src.common.topology import load_camera_topology


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_jsonable(config), handle, sort_keys=False, allow_unicode=True)
    except Exception:
        path.write_text(json.dumps(_jsonable(config), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_error_json(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def _existing_optional(name: str, value: str | None) -> str | None:
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


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def _model_arg(resolved: Any) -> str | None:
    return str(resolved.path) if getattr(resolved, "path", None) is not None else None


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest: str | None = None,
    topology: str | None = None,
    config: str | None = None,
    gt: str | None = None,
) -> Path:
    pipeline_start = time.perf_counter()
    started_at = datetime.now().astimezone()
    input_path = Path(input_dir)
    output_base = Path(output_dir)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / "pipeline" / run_ts
    detection_dir = run_dir / "1_detection"
    tracking_dir = run_dir / "2_tracking"
    pose_dir = run_dir / "3_pose"
    adl_dir = run_dir / "4_adl"
    face_dir = run_dir / "4b_face"
    reid_dir = run_dir / "5_reid"

    manifest = _existing_optional("manifest", manifest)
    topology = _existing_optional("topology", topology)
    config = _existing_optional("config", config)
    gt = _existing_optional("gt", gt)

    registry = load_model_registry(config)
    manifest_items = resolve_videos_from_manifest(input_path, Path(manifest) if manifest else None)
    topology_obj = load_camera_topology(topology)
    args_snapshot = argparse.Namespace(input=str(input_path), output=str(output_base), manifest=manifest, topology=topology, config=config, gt=gt)
    effective_config = build_effective_runtime_config(registry, manifest_items, topology_obj, args_snapshot)
    effective_config.update({
        "run_timestamp": run_ts,
        "git_commit": _git_commit(),
        "paths": {
            "input": input_path,
            "run_dir": run_dir,
            "detection": detection_dir,
            "tracking": tracking_dir,
            "pose": pose_dir,
            "adl": adl_dir,
            "face": face_dir,
            "reid": reid_dir,
            "manifest": manifest,
            "topology": topology,
            "config": config,
            "gt": gt,
        },
    })
    _write_run_config(run_dir / "run_config.yaml", effective_config)
    save_json(run_dir / "run_config.json", effective_config)
    _log("INFO", f"Pipeline run directory: {run_dir}")

    det_model = resolve_model_path(registry, "human_detection")
    track_model = resolve_model_path(registry, "human_tracking")
    pose_model = resolve_model_path(registry, "pose_estimation")
    det_cfg = get_section(registry, "human_detection")
    track_cfg = get_section(registry, "human_tracking")
    pose_cfg = get_section(registry, "pose_estimation")
    adl_cfg = get_section(registry, "adl_recognition")
    face_cfg = get_section(registry, "face")
    reid_cfg = get_section(registry, "global_reid")

    steps = [
        ("Detection", "src.human_detection.api", "process_folder", (input_path, detection_dir), {
            "model": _model_arg(det_model),
            "conf": float(det_cfg.get("conf", det_model.conf or 0.5)),
        }),
        ("Tracking", "src.human_tracking.api", "process_folder", (input_path, tracking_dir), {
            "model": _model_arg(track_model),
            "tracker": track_cfg.get("tracker_config", "bytetrack.yaml"),
            "conf": float(track_cfg.get("conf", track_model.conf or 0.5)),
            "detection_dir": detection_dir,
            "min_hits": int(track_cfg.get("min_hits", 3)),
            "max_age": int(track_cfg.get("max_age", 30)),
            "window_size": int(track_cfg.get("window_size", track_cfg.get("max_age", 30))),
        }),
        ("Pose", "src.pose_estimation.api", "process_folder", (input_path, pose_dir), {
            "model": _model_arg(pose_model),
            "conf": float(pose_cfg.get("conf", pose_model.conf or 0.5)),
            "track_dir": tracking_dir,
            "keypoint_conf": float(pose_cfg.get("keypoint_conf", 0.30)),
            "track_iou_threshold": float(pose_cfg.get("track_iou_threshold", 0.30)),
            "min_visible_keypoints": int(pose_cfg.get("min_visible_keypoints", 8)),
            "run_on_confirmed_tracks_only": bool(pose_cfg.get("run_on_confirmed_tracks_only", True)),
        }),
        ("ADL", "src.adl_recognition.api", "process_folder", (pose_dir, input_path, adl_dir), {
            "window_size": int(adl_cfg.get("window_size", 30)),
            "config": adl_cfg,
        }),
    ]

    if bool(face_cfg.get("enabled", False)):
        steps.append(("Face", "src.face.api", "process_folder", (input_path, face_dir), {
            "track_dir": tracking_dir,
            "model_config": face_cfg,
            "run_every_n_frames": int(face_cfg.get("run_every_n_frames", 10)),
        }))
    else:
        _log("WARN", "Face step disabled by config/default; skipping 4b_face.")

    steps.append(("ReID", "src.global_reid.api", "process_folder", (input_path, reid_dir), {
        "pose_dir": pose_dir,
        "adl_dir": adl_dir,
        "face_dir": face_dir if bool(face_cfg.get("enabled", False)) else None,
        "manifest": manifest_items,
        "topology": topology_obj,
        "config": reid_cfg,
    }))

    for label, module_name, function_name, args, kwargs in steps:
        _log("INFO", f"Step: {label}")
        try:
            _call(module_name, function_name, *args, **kwargs)
        except Exception as exc:
            _log("ERROR", f"{label} step failed: {exc}")
            error_dir = run_dir / label.lower()
            _write_error_json(error_dir / "error.json", {"failure_reason": ErrorCode.STEP_FAILED.value, "step": label, "error": str(exc)})

    if gt:
        _log("INFO", "Step: Evaluation")
        try:
            _call("src.evaluation.main", "evaluate_all", run_dir, Path(gt), run_dir / "evaluation")
        except Exception as exc:
            _log("ERROR", f"Evaluation failed: {exc}")
            _write_error_json(run_dir / "evaluation" / "error.json", {"failure_reason": ErrorCode.STEP_FAILED.value, "error": str(exc)})

    finished_at = datetime.now().astimezone()
    runtime = {
        "pipeline_wall_clock_runtime_sec": time.perf_counter() - pipeline_start,
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "failure_reason": ErrorCode.OK.value,
    }
    save_json(run_dir / "pipeline_runtime.json", runtime)

    _log("INFO", "Step: Benchmark")
    try:
        _call("src.pipeline.benchmark_all", "benchmark", run_dir)
    except Exception as exc:
        _log("ERROR", f"Benchmark failed: {exc}")
    _log("INFO", f"Pipeline complete: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPose TFCS-PAR terminal pipeline")
    parser.add_argument("--input", default=str(DATA_TEST_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gt", default=str(ANNOTATIONS_DIR), help="Train/val annotation root")
    args = parser.parse_args()
    print_module_console("pipeline", args)
    run_pipeline(args.input, args.output, args.manifest, args.topology, args.config, args.gt)


if __name__ == "__main__":
    main()
