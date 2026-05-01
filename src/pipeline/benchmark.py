from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src import ANNOTATIONS_DIR, print_module_console
from src.common.console import print_header, print_metric_table
from src.common.metrics import load_json, save_csv, save_json
from src.common.paths import OUTPUT_DIR, ensure_dir, resolve_path


def _metric_files(output_dir: Path) -> list[Path]:
    names = {
        "detection_metrics.json",
        "tracking_metrics.json",
        "pose_metrics.json",
        "adl_metrics.json",
    }
    return sorted(path for path in output_dir.rglob("*_metrics.json") if path.name in names)


def _module_name(path: Path) -> str:
    name = path.name.replace("_metrics.json", "")
    return name


def benchmark(output_dir: str | Path) -> dict[str, Any]:
    output_dir = resolve_path(output_dir)
    benchmark_dir = ensure_dir(output_dir / "benchmarks")
    files = _metric_files(output_dir)
    rows: list[dict[str, Any]] = []
    adl_distribution: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "total_videos": 0,
        "total_frames": 0,
        "total_persons_detected": 0,
        "total_tracks": 0,
        "total_pose_instances": 0,
        "adl_class_distribution": {},
        "output_folder_summary": str(output_dir),
    }

    videos_seen: set[str] = set()
    for metric_file in files:
        metrics = load_json(metric_file, {})
        module = _module_name(metric_file)
        video = metric_file.parent.name
        videos_seen.add(video)
        row = {"module": module, "video": video, **metrics}
        rows.append(row)
        if module == "detection":
            summary["total_frames"] += int(metrics.get("processed_frames", metrics.get("total_frames", 0)))
            summary["total_persons_detected"] += int(metrics.get("total_person_detections", 0))
        elif module == "tracking":
            summary["total_tracks"] += int(metrics.get("total_tracks", 0))
        elif module == "pose":
            summary["total_pose_instances"] += int(metrics.get("total_pose_instances", 0))
        elif module == "adl":
            adl_distribution.update(metrics.get("class_distribution", {}))

    summary["total_videos"] = len(videos_seen)
    for module in ("detection", "tracking", "pose"):
        module_rows = [row for row in rows if row["module"] == module]
        summary[f"{module}_fps"] = (
            sum(float(row.get("fps_processing", 0.0)) for row in module_rows) / len(module_rows)
            if module_rows else 0.0
        )
    adl_rows = [row for row in rows if row["module"] == "adl"]
    summary["adl_fps_equivalent"] = (
        sum(float(row.get("fps_equivalent", 0.0)) for row in adl_rows) / len(adl_rows)
        if adl_rows else 0.0
    )
    summary["adl_class_distribution"] = dict(adl_distribution)
    save_json(benchmark_dir / "summary_metrics.json", summary)
    save_csv(benchmark_dir / "summary_metrics.csv", rows)

    print_header("CPose Benchmark Summary")
    print_metric_table(summary)
    print(f"Saved summary JSON: {benchmark_dir / 'summary_metrics.json'}")
    print(f"Saved summary CSV : {benchmark_dir / 'summary_metrics.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CPose module metrics")
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--labels", default=str(ANNOTATIONS_DIR), help="Train/val label root for metric context")
    args = parser.parse_args()
    print_module_console("pipeline", args)
    benchmark(args.output)


if __name__ == "__main__":
    main()
