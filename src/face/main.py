from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

from src import ANNOTATIONS_DIR, print_module_console
from src.common.paths import MULTICAM_DIR, OUTPUT_DIR
from src.face.api import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional CPose face module")
    parser.add_argument("--input", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "4b_face"))
    parser.add_argument("--track-dir", default=str(OUTPUT_DIR / "2_tracking"))
    parser.add_argument("--run-every-n-frames", type=int, default=10)
    parser.add_argument("--config", default=None)
    parser.add_argument("--labels", default=str(ANNOTATIONS_DIR), help="Train/val label root for metric context")
    args = parser.parse_args()
    print_module_console("face", args)
    process_folder(args.input, args.output, args.track_dir, run_every_n_frames=args.run_every_n_frames)


if __name__ == "__main__":
    main()
