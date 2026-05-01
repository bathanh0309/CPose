"""make_run_summary.py — Write a human-readable Markdown run summary.

Reads ``pipeline_runtime.json`` and ``benchmark_summary.json`` from a run
directory and produces ``run_summary.md`` in the run root.

Usage:
    python -m src.reports.make_run_summary --run-dir dataset/runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _load(path: Path, default=None):
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def make_run_summary(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    runtime = _load(run_path / "pipeline_runtime.json")
    bench = _load(run_path / "benchmark_summary.json")
    run_id = run_path.name

    lines = [
        f"# CPose Run Summary — `{run_id}`",
        "",
        f"**Started:**  {runtime.get('started_at', 'N/A')}",
        f"**Finished:** {runtime.get('finished_at', 'N/A')}",
        f"**Wall clock:** {_fmt(runtime.get('pipeline_wall_clock_runtime_sec'))} s",
        "",
        "## Performance",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total videos | {bench.get('total_videos', 'N/A')} |",
        f"| Total frames | {bench.get('total_frames', 'N/A')} |",
        f"| Detection FPS | {_fmt(bench.get('detection_fps'))} |",
        f"| Tracking FPS | {_fmt(bench.get('tracking_fps'))} |",
        f"| Pose FPS | {_fmt(bench.get('pose_fps'))} |",
        f"| ADL FPS-eq | {_fmt(bench.get('adl_fps_equivalent'))} |",
        f"| ReID FPS-eq | {_fmt(bench.get('reid_fps_equivalent'))} |",
        f"| End-to-end FPS (wall) | {_fmt(bench.get('end_to_end_fps_wall_clock'))} |",
        f"| Real-time capable | {bench.get('realtime_capable', 'N/A')} |",
        f"| Metric type | {bench.get('proxy_or_gt_metric_type', 'proxy')} |",
        "",
        "## ADL Distribution",
        "",
    ]
    dist = bench.get("adl_class_distribution", {})
    total = sum(dist.values()) or 1
    lines += [f"| {label} | {count} ({count / total * 100:.1f}%) |"
              for label, count in dist.items()]

    lines += [
        "",
        "## System Info",
        "",
        f"| CPU usage | {_fmt(bench.get('cpu_usage_mean'))} % |",
        f"| RAM peak | {_fmt(bench.get('ram_usage_peak_mb'))} MB |",
        f"| GPU | {bench.get('gpu_name', 'N/A')} |",
        f"| GPU memory peak | {_fmt(bench.get('gpu_memory_peak_mb'))} MB |",
        "",
        "> Metric type: **proxy** unless `dataset/annotations/` were present.",
        "",
    ]

    out_path = run_path / "run_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved run summary: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a run summary Markdown file")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    make_run_summary(args.run_dir)


if __name__ == "__main__":
    main()
