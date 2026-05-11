"""Full CPose pipeline orchestration."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.errors import ErrorCode
from src.config import build_effective_runtime_config, get_section, load_model_registry, resolve_model_path
from src.common.json_io import save_json
from src.manifest import resolve_videos_from_manifest
from src.paths import ensure_dir, resolve_path
from src.topology import load_camera_topology
from src.stage_registry import StageCall, enabled_stages, stage_output_dirs


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


def _existing_optional(name: str, value: str | None) -> str | None:
    if not value:
        _log("WARN", f"Optional --{name} not provided; continuing with defaults/proxy behavior.")
        return None
    path = resolve_path(value)
    if not path.exists():
        _log("WARN", f"Optional --{name} path not found: {path}; continuing with defaults/proxy behavior.")
    return str(path)


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


def _safe_run_tag(run_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_id or "run")).strip("_") or "run"


def _build_stage_calls(
    input_path: Path,
    run_dir: Path,
    registry: dict[str, Any],
    manifest_items: Any,
    topology_obj: Any,
) -> tuple[StageCall, ...]:
    dirs = stage_output_dirs(run_dir)
    det_model = resolve_model_path(registry, "human_detection")
    track_model = resolve_model_path(registry, "human_tracking")
    pose_model = resolve_model_path(registry, "pose_estimation")
    det_cfg = get_section(registry, "human_detection")
    track_cfg = get_section(registry, "human_tracking")
    pose_cfg = get_section(registry, "pose_estimation")
    adl_cfg = get_section(registry, "adl_recognition")
    reid_cfg = get_section(registry, "global_reid")
    by_key = {stage.key: stage for stage in enabled_stages()}

    calls: list[StageCall] = [
        StageCall(by_key["detection"], (input_path, dirs["detection"]), {
            "model": _model_arg(det_model),
            "conf": float(det_cfg.get("conf", det_model.conf or 0.5)),
            "preview": False,
        }),
        StageCall(by_key["tracking"], (input_path, dirs["tracking"]), {
            "model": _model_arg(track_model),
            "tracker": track_cfg.get("tracker_config", "bytetrack.yaml"),
            "conf": float(track_cfg.get("conf", track_model.conf or 0.5)),
            "detection_dir": dirs["detection"],
            "min_hits": int(track_cfg.get("min_hits", 3)),
            "max_age": int(track_cfg.get("max_age", 30)),
            "window_size": int(track_cfg.get("window_size", track_cfg.get("max_age", 30))),
            "preview": False,
        }),
        StageCall(by_key["pose"], (input_path, dirs["pose"]), {
            "model": _model_arg(pose_model),
            "conf": float(pose_cfg.get("conf", pose_model.conf or 0.5)),
            "track_dir": dirs["tracking"],
            "keypoint_conf": float(pose_cfg.get("keypoint_conf", 0.30)),
            "track_iou_threshold": float(pose_cfg.get("track_iou_threshold", 0.30)),
            "min_visible_keypoints": int(pose_cfg.get("min_visible_keypoints", 8)),
            "run_on_confirmed_tracks_only": bool(pose_cfg.get("run_on_confirmed_tracks_only", True)),
            "preview": False,
        }),
        StageCall(by_key["adl"], (dirs["pose"], input_path, dirs["adl"]), {
            "window_size": int(adl_cfg.get("window_size", 30)),
            "config": adl_cfg,
            "preview": False,
        }),
    ]

    calls.append(StageCall(by_key["reid"], (input_path, dirs["reid"]), {
        "pose_dir": dirs["pose"],
        "adl_dir": dirs["adl"],
        "face_dir": None,
        "manifest": manifest_items,
        "topology": topology_obj,
        "config": reid_cfg,
        "preview": True,
    }))
    return tuple(calls)


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest: str | None = None,
    topology: str | None = None,
    config: str | None = None,
    gt: str | None = None,
    run_id: str = "run",
    models: str | None = None,
) -> Path:
    """Run all enabled CPose stages and return the created run directory."""

    pipeline_start = time.perf_counter()
    started_at = datetime.now().astimezone()
    input_path = resolve_path(input_dir)
    output_base = ensure_dir(output_dir)
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = _safe_run_tag(run_id)
    run_dir = output_base / f"{run_ts}_{safe_tag}"
    dirs = stage_output_dirs(run_dir)

    manifest = _existing_optional("manifest", manifest)
    topology = _existing_optional("topology", topology)
    config = _existing_optional("config", config)
    model_registry_path = _existing_optional("models", models or config)
    gt = _existing_optional("gt", gt)

    registry = load_model_registry(model_registry_path)
    manifest_items = resolve_videos_from_manifest(input_path, Path(manifest) if manifest else None)
    topology_obj = load_camera_topology(topology)
    args_snapshot = argparse.Namespace(
        input=str(input_path),
        output=str(output_base),
        manifest=manifest,
        topology=topology,
        config=config,
        models=model_registry_path,
        gt=gt,
        run_id=safe_tag,
    )
    effective_config = build_effective_runtime_config(registry, manifest_items, topology_obj, args_snapshot)
    effective_config.update({
        "run_timestamp": run_ts,
        "git_commit": _git_commit(),
        "paths": {
            "input": input_path,
            "run_dir": run_dir,
            **dirs,
            "manifest": manifest,
            "topology": topology,
            "config": config,
            "models": model_registry_path,
            "gt": gt,
        },
    })
    _write_run_config(run_dir / "run_config.yaml", effective_config)
    save_json(run_dir / "run_config.json", effective_config)
    _log("INFO", f"Pipeline run directory: {run_dir}")

    for call in _build_stage_calls(input_path, run_dir, registry, manifest_items, topology_obj):
        _log("INFO", f"Step: {call.spec.name}")
        try:
            _call(call.spec.module_name, call.spec.function_name, *call.args, **call.kwargs)
        except Exception as exc:
            _log("ERROR", f"{call.spec.name} step failed: {exc}")
            error_dir = run_dir / (call.spec.output_dir_name or call.spec.key)
            save_json(error_dir / "error.json", {"failure_reason": ErrorCode.STEP_FAILED.value, "step": call.spec.name, "error": str(exc)})

    if gt:
        _log("INFO", "Step: Evaluation")
        try:
            _call("src.evaluation.main", "evaluate_all", run_dir, Path(gt), run_dir / "07_evaluation")
        except Exception as exc:
            _log("ERROR", f"Evaluation failed: {exc}")
            save_json(run_dir / "07_evaluation" / "error.json", {"failure_reason": ErrorCode.STEP_FAILED.value, "error": str(exc)})

    finished_at = datetime.now().astimezone()
    save_json(run_dir / "pipeline_runtime.json", {
        "pipeline_wall_clock_runtime_sec": time.perf_counter() - pipeline_start,
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "failure_reason": ErrorCode.OK.value,
    })

    _log("INFO", "Step: Benchmark")
    try:
        _call("src.pipeline.benchmark_all", "benchmark", run_dir)
    except Exception as exc:
        _log("ERROR", f"Benchmark failed: {exc}")
    _log("INFO", f"Pipeline complete: {run_dir}")
    return run_dir


__all__ = ["run_pipeline"]
