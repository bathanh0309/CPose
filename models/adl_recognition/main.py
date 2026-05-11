"""CLI entrypoint for CPose Module 4: ADL recognition."""
from __future__ import annotations

import argparse

from src import ANNOTATIONS_DIR, print_module_console
from src.config import get_research_section
from src.paths import MULTICAM_DIR, OUTPUT_DIR
from models.adl_recognition.api import process_folder


def main() -> None:
    phase3_config = get_research_section("phase3")
    parser = argparse.ArgumentParser(description="CPose ADL recognition module")
    parser.add_argument("--pose-dir", default=str(OUTPUT_DIR / "03_pose"))
    parser.add_argument("--video-dir", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "04_adl"))
    parser.add_argument("--window-size", type=int, default=int(phase3_config.get("window_size", 30)))
    parser.add_argument("--config", default=None)
    parser.add_argument("--labels", default=str(ANNOTATIONS_DIR), help="Train/val label root for metric context")
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--preview", dest="preview", action="store_true", default=False, help="Show processed video while running")
    preview_group.add_argument("--no-preview", dest="preview", action="store_false", help="Disable processed-video preview")
    args = parser.parse_args()
    print_module_console("adl_recognition", args)
    if args.config:
        print(f"[WARN] --config is accepted for CLI compatibility; module uses local defaults: {args.config}")
    process_folder(args.pose_dir, args.video_dir, args.output, args.window_size, preview=args.preview)


if __name__ == "__main__":
    main()
