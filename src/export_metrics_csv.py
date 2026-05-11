"""export_metrics_csv.py — Export all module metrics from a run dir as a flat CSV.

Walks the run directory, collects every ``*_metrics.json`` file, and writes a
single ``all_metrics_flat.csv`` to ``<run_dir>/08_paper_report/``.

Usage:
    python -m src.reports.export_metrics_csv --run-dir dataset/runs/<run_id>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

_METRIC_GLOBS = [
    "detection_metrics.json",
    "tracking_metrics.json",
    "pose_metrics.json",
    "adl_metrics.json",
    "reid_metrics.json",
]


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _flatten(data: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict into dot-notation keys."""
    out = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full_key))
        elif isinstance(value, (list, tuple)):
            out[full_key] = json.dumps(value, ensure_ascii=False)
        else:
            out[full_key] = value
    return out


def export_metrics_csv(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    report_dir = run_path / "08_paper_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for glob in _METRIC_GLOBS:
        for path in sorted(run_path.rglob(glob)):
            try:
                data = _load(path)
            except Exception as exc:
                print(f"[WARN] Could not load {path}: {exc}")
                continue
            flat = _flatten(data)
            flat["_source_file"] = str(path.relative_to(run_path))
            flat["_module"] = glob.replace("_metrics.json", "")
            rows.append(flat)

    if not rows:
        print("[WARN] No metric files found. Nothing exported.")
        return

    # Build a union of all keys (some modules have different keys)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    out_path = report_dir / "all_metrics_flat.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "N/A") for k in all_keys})

    print(f"Exported {len(rows)} metric records to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all module metrics as a flat CSV")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    export_metrics_csv(args.run_dir)


if __name__ == "__main__":
    main()
