"""CLI entrypoint for CPose Module 5: global ReID."""
from __future__ import annotations

import argparse

from src import ANNOTATIONS_DIR, print_module_console
from src.paths import MULTICAM_DIR, OUTPUT_DIR
from models.global_reid.api import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="CPose cross-camera ReID module")
    parser.add_argument("--input", default=str(MULTICAM_DIR), help="Folder containing multicam mp4 clips")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "05_global_reid"), help="Output folder")
    parser.add_argument("--pose-dir", default=str(OUTPUT_DIR / "03_pose"), help="Module 3 pose output folder")
    parser.add_argument("--adl-dir", default=str(OUTPUT_DIR / "04_adl"), help="Module 4 ADL output folder")
    parser.add_argument("--face-dir", default=None, help="Optional face output folder")
    parser.add_argument("--video-dir", default=None, help="Alias for --input")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--labels", default=str(ANNOTATIONS_DIR), help="Train/val label root for metric context")
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--preview", dest="preview", action="store_true", default=False, help="Show processed video while running")
    preview_group.add_argument("--no-preview", dest="preview", action="store_false", help="Disable processed-video preview")
    args = parser.parse_args()
    print_module_console("global_reid", args)
    process_folder(
        args.video_dir or args.input,
        args.output,
        args.pose_dir,
        args.adl_dir,
        args.face_dir,
        manifest=args.manifest,
        topology=args.topology,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
