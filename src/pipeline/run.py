"""
Full CPose TFCS-PAR Pipeline
==============================
Chạy tuần tự: Detection → Tracking → Pose → ADL → ReID (cross-camera Global ID)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from pathlib import Path

from src import ANNOTATIONS_DIR, DEFAULT_CONFIG, print_module_console
from src.common.console import print_header
from src.common.paths import MULTICAM_DIR, OUTPUT_DIR, ensure_dir, resolve_path
from src.human_detection.api import process_folder as process_detection
from src.human_tracking.api import process_folder as process_tracking
from src.pose_estimation.api import process_folder as process_pose
from src.adl_recognition.api import process_folder as process_adl
from src.global_reid.api import process_folder as process_reid


def run_pipeline(input_dir: str | Path, output_dir: str | Path) -> None:
    input_dir = resolve_path(input_dir)
    output_dir = ensure_dir(output_dir)
    detection_dir = output_dir / "1_detection"
    tracking_dir = output_dir / "2_tracking"
    pose_dir = output_dir / "3_pose"
    adl_dir = output_dir / "4_adl"
    reid_dir = output_dir / "5_reid"

    print_header("CPose TFCS-PAR Full Pipeline")
    print("Step 1/5: Person detection (no ID labels)")
    process_detection(input_dir, detection_dir)

    print("\nStep 2/5: Person tracking (local clip IDs only)")
    process_tracking(input_dir, tracking_dir, detection_dir=detection_dir)

    print("\nStep 3/5: Pose estimation (no Global ID labels)")
    process_pose(input_dir, pose_dir, track_dir=tracking_dir)

    print("\nStep 4/5: ADL recognition (no Global ID labels)")
    process_adl(pose_dir, input_dir, adl_dir)

    print("\nStep 5/5: Cross-camera ReID — assigns Global IDs (GID-001, GID-002, ...)")
    process_reid(input_dir, reid_dir, pose_dir=pose_dir, adl_dir=adl_dir)

    print_header("PIPELINE COMPLETE")
    print(f"Output folder: {output_dir}")
    print(f"  1_detection  -> {detection_dir}")
    print(f"  2_tracking   -> {tracking_dir}")
    print(f"  3_pose       -> {pose_dir}")
    print(f"  4_adl        -> {adl_dir}")
    print(f"  5_reid       -> {reid_dir}  [Global IDs GID-001... rendered here]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full CPose TFCS-PAR pipeline")
    parser.add_argument("--input", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gt", default=str(ANNOTATIONS_DIR), help="Train/val annotation root")
    args = parser.parse_args()
    print_module_console("pipeline", args)
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
