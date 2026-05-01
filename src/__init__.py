"""Shared CPose module defaults and terminal helpers.

Each package under :mod:`src` owns one pipeline responsibility.  The entry
points import this module only for common defaults, direct-script support, and
consistent terminal output.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TEST_DIR = PROJECT_ROOT / "data-test"
DATASET_DIR = PROJECT_ROOT / "dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
OUTPUT_DIR = DATASET_DIR / "outputs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_registry.demo_i5.yaml"


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    name: str
    responsibility: str
    raw_input: str
    processed_output: str
    default_output: Path | None = None


MODULES: dict[str, ModuleSpec] = {
    "human_detection": ModuleSpec(
        name="Human Detection",
        responsibility="Detect person bounding boxes from raw test videos.",
        raw_input="data-test/*.mp4",
        processed_output="detections.json, detection_overlay.mp4, detection_metrics.json",
        default_output=OUTPUT_DIR / "1_detection",
    ),
    "human_tracking": ModuleSpec(
        name="Human Tracking",
        responsibility="Assign local track_id values inside each camera clip.",
        raw_input="data-test/*.mp4 plus optional dataset/outputs/1_detection",
        processed_output="tracks.json, tracking_overlay.mp4, tracking_metrics.json",
        default_output=OUTPUT_DIR / "2_tracking",
    ),
    "pose_estimation": ModuleSpec(
        name="Pose Estimation",
        responsibility="Estimate COCO-17 keypoints and attach local track_id when available.",
        raw_input="data-test/*.mp4 plus dataset/outputs/2_tracking",
        processed_output="keypoints.json, pose_overlay.mp4, pose_metrics.json",
        default_output=OUTPUT_DIR / "3_pose",
    ),
    "adl_recognition": ModuleSpec(
        name="ADL Recognition",
        responsibility="Classify rule-based ADL labels from processed pose sequences.",
        raw_input="dataset/outputs/3_pose plus data-test/*.mp4",
        processed_output="adl_events.json, adl_overlay.mp4, adl_metrics.json",
        default_output=OUTPUT_DIR / "4_adl",
    ),
    "face": ModuleSpec(
        name="Face",
        responsibility="Extract optional face evidence for downstream global ReID.",
        raw_input="data-test/*.mp4 plus dataset/outputs/2_tracking",
        processed_output="face_events.json, face_metrics.json",
        default_output=OUTPUT_DIR / "4b_face",
    ),
    "global_reid": ModuleSpec(
        name="Global ReID",
        responsibility="Fuse pose, ADL, face, time, and topology cues into cross-camera global_id values.",
        raw_input="data-test/*.mp4 plus processed pose/ADL/face folders",
        processed_output="reid_tracks.json, reid_overlay.mp4, global_person_table.json, reid_metrics.json",
        default_output=OUTPUT_DIR / "5_reid",
    ),
    "evaluation": ModuleSpec(
        name="Evaluation",
        responsibility="Compare outputs with train/val labels when ground truth exists; otherwise keep proxy metrics.",
        raw_input="dataset/outputs plus dataset/annotations",
        processed_output="evaluation_summary.json, evaluation_summary.csv",
        default_output=OUTPUT_DIR / "evaluation",
    ),
    "pipeline": ModuleSpec(
        name="Pipeline",
        responsibility="Run all CPose modules in order on the data-test test split.",
        raw_input="data-test/*.mp4 plus optional configs and dataset/annotations",
        processed_output="timestamped dataset/outputs/pipeline run with overlays, JSON logs, metrics, and benchmark",
        default_output=OUTPUT_DIR,
    ),
    "live_pipeline": ModuleSpec(
        name="Live Pipeline",
        responsibility="Unified frame-by-frame pipeline: Detection + Tracking + Pose + ADL in one preview window.",
        raw_input="data-test/*.mp4",
        processed_output="live_overlay.mp4, live_records.json per video in dataset/outputs/live/",
        default_output=OUTPUT_DIR / "live",
    ),
}


def configure_direct_run(file: str | Path) -> None:
    """Make ``python src/<module>/main.py`` behave like ``python -m``.

    IDE run buttons often execute a file directly.  In that mode Python may not
    include the project root on ``sys.path``, so absolute imports such as
    ``src.common.paths`` can fail.
    """

    root = Path(file).resolve().parents[2]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def module_spec(module_key: str) -> ModuleSpec:
    try:
        return MODULES[module_key]
    except KeyError as exc:
        known = ", ".join(sorted(MODULES))
        raise KeyError(f"Unknown CPose module '{module_key}'. Known modules: {known}") from exc


def print_module_console(module_key: str, args: Any) -> None:
    """Print raw input context and processed-output context for a module run."""

    spec = module_spec(module_key)
    raw_rows = {
        "Module": spec.name,
        "Responsibility": spec.responsibility,
        "Test input": getattr(args, "input", None) or getattr(args, "video_dir", None) or DATA_TEST_DIR,
        "Train/val labels": getattr(args, "labels", None) or getattr(args, "gt", None) or ANNOTATIONS_DIR,
        "Config": getattr(args, "config", None) or DEFAULT_CONFIG,
    }
    processed_rows = {
        "Expected raw source": spec.raw_input,
        "Processed output": spec.processed_output,
        "Output folder": getattr(args, "output", None) or getattr(args, "out", None) or spec.default_output,
    }
    for attr in ("detection_dir", "track_dir", "pose_dir", "adl_dir", "face_dir", "outputs", "manifest", "topology"):
        value = getattr(args, attr, None)
        if value:
            processed_rows[attr.replace("_", " ").title()] = value

    _print_box("RAW INPUT CONSOLE", raw_rows)
    _print_box("PROCESSED MODULE CONSOLE", processed_rows)


def _print_box(title: str, rows: dict[str, Any]) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    width = max(len(key) for key in rows)
    for key, value in rows.items():
        print(f"{key:<{width}} : {value}")
