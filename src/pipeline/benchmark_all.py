from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _save_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({k: v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False) for k, v in row.items()})


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cpu_usage_mean": None, "ram_usage_peak_mb": None, "gpu_name": None, "gpu_memory_peak_mb": None}
    try:
        import psutil

        info["cpu_usage_mean"] = psutil.cpu_percent(interval=0.1)
        info["ram_usage_peak_mb"] = round(psutil.virtual_memory().used / (1024 * 1024), 2)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(index)
            info["gpu_memory_peak_mb"] = round(torch.cuda.max_memory_allocated(index) / (1024 * 1024), 2)
    except Exception:
        pass
    return info


def benchmark(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics = {
        "detection": [_load_json(path, {}) for path in run_path.rglob("detection_metrics.json")],
        "tracking": [_load_json(path, {}) for path in run_path.rglob("tracking_metrics.json")],
        "pose": [_load_json(path, {}) for path in run_path.rglob("pose_metrics.json")],
        "adl": [_load_json(path, {}) for path in run_path.rglob("adl_metrics.json")],
        "reid": [_load_json(path, {}) for path in run_path.rglob("reid_metrics.json")],
    }
    evaluation = _load_json(run_path / "evaluation" / "evaluation_summary.json", {})
    pipeline_runtime = _load_json(run_path / "pipeline_runtime.json", {})
    videos = {
        Path(str(row.get("input_video"))).stem
        for rows in metrics.values()
        for row in rows
        if row.get("input_video")
    }
    adl_distribution: Counter[str] = Counter()
    for row in metrics["adl"]:
        adl_distribution.update(row.get("class_distribution", {}))
    end_frames = sum(int(row.get("processed_frames", row.get("total_frames", 0))) for row in metrics["detection"])
    elapsed_values = [float(row.get("elapsed_sec", 0.0)) for rows in metrics.values() for row in rows if row.get("elapsed_sec") is not None]
    offline_elapsed = sum(elapsed_values) if elapsed_values else None
    wall_elapsed = pipeline_runtime.get("pipeline_wall_clock_runtime_sec")
    metric_types = [row.get("metric_type") for rows in metrics.values() for row in rows] + [evaluation.get("metric_type")]
    input_fps_values = [float(row.get("video_fps", row.get("input_fps", 0.0))) for rows in metrics.values() for row in rows if row.get("video_fps") or row.get("input_fps")]
    avg_input_fps = _mean(input_fps_values)
    fps_wall = (end_frames / float(wall_elapsed)) if end_frames and wall_elapsed else None
    failure_counter: Counter[str] = Counter()
    for rows in metrics.values():
        for row in rows:
            failure_counter.update(row.get("failure_reason_distribution", {}))
            if row.get("failure_reason"):
                failure_counter.update([row["failure_reason"]])
    summary: dict[str, Any] = {
        "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_videos": len(videos) if videos else None,
        "total_frames": end_frames if end_frames else None,
        "detection_fps": _mean([float(row["fps_processing"]) for row in metrics["detection"] if row.get("fps_processing") is not None]),
        "tracking_fps": _mean([float(row["fps_processing"]) for row in metrics["tracking"] if row.get("fps_processing") is not None]),
        "pose_fps": _mean([float(row["fps_processing"]) for row in metrics["pose"] if row.get("fps_processing") is not None]),
        "adl_fps_equivalent": _mean([float(row["fps_equivalent"]) for row in metrics["adl"] if row.get("fps_equivalent") is not None]),
        "reid_fps_equivalent": _mean([float(row["fps_processing"]) for row in metrics["reid"] if row.get("fps_processing") is not None]),
        "offline_module_sum_runtime_sec": offline_elapsed,
        "pipeline_wall_clock_runtime_sec": wall_elapsed,
        "end_to_end_fps_offline": (end_frames / offline_elapsed) if end_frames and offline_elapsed else None,
        "end_to_end_fps_wall_clock": fps_wall,
        "realtime_capable": bool(fps_wall is not None and avg_input_fps is not None and fps_wall >= avg_input_fps) if avg_input_fps else None,
        "average_latency_per_frame_ms": None,
        "total_detections": sum(int(row.get("total_person_detections", 0)) for row in metrics["detection"]) or None,
        "total_tracks": sum(int(row.get("total_tracks", 0)) for row in metrics["tracking"]) or None,
        "total_pose_instances": sum(int(row.get("total_pose_instances", 0)) for row in metrics["pose"]) or None,
        "adl_class_distribution": dict(adl_distribution),
        "unknown_rate": evaluation.get("modules", {}).get("adl", {}).get("unknown_rate"),
        "global_id_count": evaluation.get("modules", {}).get("reid", {}).get("global_id_count"),
        "proxy_or_gt_metric_type": "ground_truth" if "ground_truth" in metric_types else "proxy",
        "failure_reason_distribution": dict(failure_counter),
        "failure_reason": "OK",
    }
    summary.update(_system_info())
    _save_json(run_path / "benchmark_summary.json", summary)
    _save_csv(run_path / "benchmark_summary.csv", summary)
    print("[INFO] CPose Benchmark Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CPose benchmark metrics")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    benchmark(args.run_dir)


if __name__ == "__main__":
    main()
