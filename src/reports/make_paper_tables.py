"""make_paper_tables.py — Generate paper-ready CSV and Markdown tables from a CPose run directory.

Usage:
    python -m src.reports.make_paper_tables --run-dir dataset/runs/<run_id>

Output (written to ``<run_dir>/08_paper_report/``):
    paper_metrics_summary.md
    table_module_runtime.csv
    table_detection_results.csv
    table_tracking_results.csv
    table_pose_results.csv
    table_adl_results.csv
    table_reid_results.csv
    figure_list.md

Rules (per CLAUDE.md §13):
- Only summary numbers, no per-frame data.
- Always include a metric_type column.
- N/A for any metric that could not be computed — never fabricate.
- Directly paste-able into LaTeX tabular or Markdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value: Any, precision: int = 2) -> str:
    """Format a numeric value or return N/A."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(row) + " |\n")


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def _collect_module_metrics(run_dir: Path, glob: str) -> list[dict]:
    return [_load(p, {}) for p in sorted(run_dir.rglob(glob))]


def _collect_video_stems(run_dir: Path) -> list[str]:
    """Return all video stems that have at least one module output."""
    stems: set[str] = set()
    for pattern in ("**/detections.json", "**/tracks.json", "**/keypoints.json"):
        for p in run_dir.rglob(pattern):
            stems.add(p.parent.name)
    return sorted(stems)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _build_runtime_table(run_dir: Path) -> list[dict[str, Any]]:
    """table_module_runtime.csv — per CLAUDE.md §13."""
    modules = [
        ("detection",   "detection_metrics.json",  "fps_processing",   "avg_latency_ms_per_frame",  "processed_frames",     "total_person_detections"),
        ("tracking",    "tracking_metrics.json",    "fps_processing",   "avg_latency_ms_per_frame",  "total_frames",         "total_tracks"),
        ("pose",        "pose_metrics.json",        "fps_processing",   "avg_latency_ms_per_frame",  "total_frames",         "total_pose_instances"),
        ("adl",         "adl_metrics.json",         "fps_equivalent",   None,                         "total_frames",         "total_adl_events"),
        ("global_reid", "reid_metrics.json",        "fps_processing",   "avg_latency_ms_per_frame",  "total_frames",         "unique_global_ids"),
    ]
    rows = []
    for module, glob, fps_key, latency_key, frames_key, instances_key in modules:
        records = _collect_module_metrics(run_dir, glob)
        if not records:
            rows.append({
                "module": module,
                "total_frames": "N/A",
                "total_instances": "N/A",
                "fps": "N/A",
                "latency_ms": "N/A",
                "metric_type": "proxy",
            })
            continue
        fps_vals = [float(r[fps_key]) for r in records if r.get(fps_key) is not None]
        lat_vals = [float(r[latency_key]) for r in records if latency_key and r.get(latency_key) is not None]
        frames_vals = [int(r.get(frames_key, 0)) for r in records]
        inst_vals = [int(r.get(instances_key, 0)) for r in records]
        metric_types = [r.get("metric_type", "proxy") for r in records]
        rows.append({
            "module": module,
            "total_frames": sum(frames_vals) or "N/A",
            "total_instances": sum(inst_vals) or "N/A",
            "fps": _fmt(sum(fps_vals) / len(fps_vals) if fps_vals else None),
            "latency_ms": _fmt(sum(lat_vals) / len(lat_vals) if lat_vals else None),
            "metric_type": "ground_truth" if "ground_truth" in metric_types else "proxy",
        })
    return rows


def _build_detection_table(run_dir: Path) -> list[dict[str, Any]]:
    records = _collect_module_metrics(run_dir, "detection_metrics.json")
    rows = []
    for r in records:
        rows.append({
            "camera": r.get("camera_id", "N/A"),
            "total_frames": r.get("processed_frames", r.get("total_frames", "N/A")),
            "total_detections": r.get("total_person_detections", "N/A"),
            "avg_persons_per_frame": _fmt(r.get("avg_persons_per_frame")),
            "avg_confidence": _fmt(r.get("avg_confidence")),
            "fps": _fmt(r.get("fps_processing")),
            "latency_ms": _fmt(r.get("avg_latency_ms_per_frame")),
            "precision": r.get("precision", "N/A"),
            "recall": r.get("recall", "N/A"),
            "f1": r.get("f1", "N/A"),
            "map50": r.get("mAP50", "N/A"),
            "metric_type": r.get("metric_type", "proxy"),
        })
    return rows


def _build_tracking_table(run_dir: Path) -> list[dict[str, Any]]:
    records = _collect_module_metrics(run_dir, "tracking_metrics.json")
    rows = []
    for r in records:
        rows.append({
            "camera": r.get("camera_id", "N/A"),
            "total_frames": r.get("total_frames", "N/A"),
            "total_tracks": r.get("total_tracks", "N/A"),
            "mean_track_age": _fmt(r.get("mean_track_age")),
            "confirmed_track_ratio": _fmt(r.get("confirmed_track_ratio")),
            "fragment_proxy": r.get("fragment_proxy", "N/A"),
            "fps": _fmt(r.get("fps_processing")),
            "latency_ms": _fmt(r.get("avg_latency_ms_per_frame")),
            "idf1": r.get("idf1", "N/A"),
            "id_switches": r.get("id_switches", "N/A"),
            "hota": r.get("hota", "N/A"),
            "metric_type": r.get("metric_type", "proxy"),
        })
    return rows


def _build_pose_table(run_dir: Path) -> list[dict[str, Any]]:
    records = _collect_module_metrics(run_dir, "pose_metrics.json")
    rows = []
    for r in records:
        rows.append({
            "camera": r.get("camera_id", "N/A"),
            "total_frames": r.get("total_frames", "N/A"),
            "total_pose_instances": r.get("total_pose_instances", "N/A"),
            "mean_keypoint_confidence": _fmt(r.get("mean_keypoint_confidence")),
            "visible_keypoint_ratio": _fmt(r.get("visible_keypoint_ratio")),
            "missing_keypoint_rate": _fmt(r.get("missing_keypoint_rate")),
            "fps": _fmt(r.get("fps_processing")),
            "latency_ms": _fmt(r.get("avg_latency_ms_per_frame")),
            "pck_01": r.get("pck_01", "N/A"),
            "pck_005": r.get("pck_005", "N/A"),
            "metric_type": r.get("metric_type", "proxy"),
        })
    return rows


def _build_adl_table(run_dir: Path) -> list[dict[str, Any]]:
    records = _collect_module_metrics(run_dir, "adl_metrics.json")
    rows = []
    for r in records:
        dist = r.get("class_distribution", {})
        total = r.get("total_adl_events", 0) or 1
        rows.append({
            "camera": r.get("camera_id", "N/A"),
            "total_adl_events": r.get("total_adl_events", "N/A"),
            "unknown_rate": _fmt(r.get("unknown_rate")),
            "avg_confidence": _fmt(r.get("avg_confidence")),
            "fps_equivalent": _fmt(r.get("fps_equivalent")),
            "standing_pct": _fmt(dist.get("standing", 0) / total * 100),
            "sitting_pct": _fmt(dist.get("sitting", 0) / total * 100),
            "walking_pct": _fmt(dist.get("walking", 0) / total * 100),
            "lying_down_pct": _fmt(dist.get("lying_down", 0) / total * 100),
            "falling_pct": _fmt(dist.get("falling", 0) / total * 100),
            "accuracy": r.get("accuracy", "N/A"),
            "macro_f1": r.get("macro_f1", "N/A"),
            "metric_type": r.get("metric_type", "proxy"),
        })
    return rows


def _build_reid_table(run_dir: Path) -> list[dict[str, Any]]:
    records = _collect_module_metrics(run_dir, "reid_metrics.json")
    # Filter out the top-level summary (which has different keys)
    clip_records = [r for r in records if r.get("clip")]
    summary_records = [r for r in records if not r.get("clip")]
    rows = []
    for r in clip_records:
        rows.append({
            "clip": r.get("clip", "N/A"),
            "camera": r.get("camera_id", "N/A"),
            "unique_global_ids": r.get("unique_global_ids", "N/A"),
            "strong_match_count": r.get("strong_match_count", "N/A"),
            "soft_match_count": r.get("soft_match_count", "N/A"),
            "new_id_count": r.get("new_id_count", "N/A"),
            "topology_conflict_count": r.get("topology_conflict_count", "N/A"),
            "avg_score_total": _fmt(r.get("avg_score_total")),
            "fps": _fmt(r.get("fps_processing")),
            "global_id_accuracy": r.get("global_id_accuracy", "N/A"),
            "cross_camera_idf1": r.get("cross_camera_idf1", "N/A"),
            "false_split_rate": r.get("false_split_rate", "N/A"),
            "false_merge_rate": r.get("false_merge_rate", "N/A"),
            "metric_type": r.get("metric_type", "proxy"),
        })
    if not rows and summary_records:
        r = summary_records[0]
        rows.append({
            "clip": "ALL",
            "camera": "ALL",
            "unique_global_ids": r.get("total_global_ids", "N/A"),
            "strong_match_count": r.get("strong_match_count", "N/A"),
            "soft_match_count": r.get("soft_match_count", "N/A"),
            "new_id_count": r.get("new_id_count", "N/A"),
            "topology_conflict_count": r.get("topology_conflict_count", "N/A"),
            "avg_score_total": _fmt(r.get("avg_score_total")),
            "fps": "N/A",
            "global_id_accuracy": "N/A",
            "cross_camera_idf1": "N/A",
            "false_split_rate": "N/A",
            "false_merge_rate": "N/A",
            "metric_type": r.get("metric_type", "proxy"),
        })
    return rows


# ---------------------------------------------------------------------------
# Summary narrative
# ---------------------------------------------------------------------------

def _write_paper_summary(report_dir: Path, run_dir: Path, runtime_rows: list[dict]) -> None:
    run_id = run_dir.name
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# CPose Paper Metrics Summary",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {timestamp}",
        f"",
        f"> All metrics labelled `proxy` were computed without ground-truth annotations.",
        f"> Metrics labelled `ground_truth` used annotations from `dataset/annotations/`.",
        f"> Never present proxy metrics as accuracy.",
        f"",
        f"## Module Runtime Performance",
        f"",
        f"| Module | Total Frames | Instances | FPS | Latency (ms) | Metric Type |",
        f"|---|---|---|---|---|---|",
    ]
    for r in runtime_rows:
        lines.append(
            f"| {r['module']} | {r['total_frames']} | {r['total_instances']} "
            f"| {r['fps']} | {r['latency_ms']} | {r['metric_type']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- See `table_detection_results.csv`, `table_tracking_results.csv`, "
          "`table_pose_results.csv`, `table_adl_results.csv`, `table_reid_results.csv` "
          "for per-camera detail.",
        "- See `figure_list.md` for overlay and comparison video paths.",
        "- Do not copy proxy values into a paper table without labelling them as proxy.",
        "",
    ]
    (report_dir / "paper_metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_figure_list(report_dir: Path, run_dir: Path) -> None:
    lines = ["# Figure List\n"]
    for pattern, label in [
        ("**/detection_overlay.mp4", "Detection Overlay"),
        ("**/tracking_overlay.mp4", "Tracking Overlay"),
        ("**/pose_overlay.mp4", "Pose Overlay"),
        ("**/adl_overlay.mp4", "ADL Overlay"),
        ("**/reid_overlay.mp4", "ReID Overlay"),
        ("**/*_raw_vs_*.mp4", "Comparison"),
    ]:
        for p in sorted(run_dir.rglob(pattern)):
            rel = p.relative_to(run_dir)
            lines.append(f"- **{label}**: `{rel}`")
    (report_dir / "figure_list.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def make_paper_tables(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"[ERROR] Run directory not found: {run_path}")
        return

    report_dir = run_path / "08_paper_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CPose Paper Table Generator")
    print("=" * 60)
    print(f"Run dir    : {run_path}")
    print(f"Report dir : {report_dir}")

    runtime_rows = _build_runtime_table(run_path)
    _write_csv(report_dir / "table_module_runtime.csv", runtime_rows)
    print("Saved : table_module_runtime.csv")

    det_rows = _build_detection_table(run_path)
    _write_csv(report_dir / "table_detection_results.csv", det_rows)
    print("Saved : table_detection_results.csv")

    trk_rows = _build_tracking_table(run_path)
    _write_csv(report_dir / "table_tracking_results.csv", trk_rows)
    print("Saved : table_tracking_results.csv")

    pose_rows = _build_pose_table(run_path)
    _write_csv(report_dir / "table_pose_results.csv", pose_rows)
    print("Saved : table_pose_results.csv")

    adl_rows = _build_adl_table(run_path)
    _write_csv(report_dir / "table_adl_results.csv", adl_rows)
    print("Saved : table_adl_results.csv")

    reid_rows = _build_reid_table(run_path)
    _write_csv(report_dir / "table_reid_results.csv", reid_rows)
    print("Saved : table_reid_results.csv")

    _write_paper_summary(report_dir, run_path, runtime_rows)
    print("Saved : paper_metrics_summary.md")

    _write_figure_list(report_dir, run_path)
    print("Saved : figure_list.md")

    print("=" * 60)
    print(f"All paper tables written to: {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready tables from a CPose run directory"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a CPose run directory (e.g., dataset/runs/2026-05-01_baseline)",
    )
    args = parser.parse_args()
    make_paper_tables(args.run_dir)


if __name__ == "__main__":
    main()
