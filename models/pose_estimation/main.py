"""CLI entrypoint for CPose Module 3: pose estimation."""
from __future__ import annotations

import argparse

from src import ANNOTATIONS_DIR, print_module_console
from src.paths import MULTICAM_DIR, OUTPUT_DIR
from models.pose_estimation.api import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="CPose pose estimation module")
    parser.add_argument("--input", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "03_pose"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--track-dir", "--tracks", dest="track_dir", default=str(OUTPUT_DIR / "02_tracking"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--labels", default=str(ANNOTATIONS_DIR), help="Train/val label root for metric context")
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--preview", dest="preview", action="store_true", default=False, help="Show processed video while running")
    preview_group.add_argument("--no-preview", dest="preview", action="store_false", help="Disable processed-video preview")
    args = parser.parse_args()
    print_module_console("pose_estimation", args)
    if args.config:
        print(f"[WARN] --config is accepted for CLI compatibility; module uses local defaults: {args.config}")
    process_folder(args.input, args.output, args.model, args.conf, args.track_dir, preview=args.preview)


if __name__ == "__main__":
    main()
