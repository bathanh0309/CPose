"""CLI entrypoint for CPose Module 1: detection."""
from __future__ import annotations

import argparse

from src import ANNOTATIONS_DIR, print_module_console
from src.common.paths import MULTICAM_DIR, OUTPUT_DIR
from src.modules.detection.api import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="CPose person detection module")
    parser.add_argument("--input", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "01_detection"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--labels", default=str(ANNOTATIONS_DIR), help="Train/val label root for metric context")
    parser.add_argument("--no-crops", action="store_true", help="Skip saving detection crops")
    parser.add_argument("--make-comparison", action="store_true", help="Generate raw-vs-processed comparison videos")
    parser.add_argument("--compare-count", type=int, default=2, help="Number of videos to compare (default: 2)")
    parser.add_argument("--comparison-dir", default=None, help="Output dir for comparison videos")
    args = parser.parse_args()
    print_module_console("human_detection", args)
    if args.config:
        print(f"[WARN] --config is accepted for CLI compatibility; module uses local defaults: {args.config}")
    process_folder(
        args.input,
        args.output,
        args.model,
        args.conf,
        save_crops=not args.no_crops,
        make_comparison=args.make_comparison,
        compare_count=args.compare_count,
        comparison_dir=args.comparison_dir,
    )


if __name__ == "__main__":
    main()
