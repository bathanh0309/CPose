from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

from src.adl_recognition.api import process_folder
from src.common.config import get_research_section
from src.common.paths import MULTICAM_DIR, OUTPUT_DIR


def main() -> None:
    phase3_config = get_research_section("phase3")
    parser = argparse.ArgumentParser(description="CPose ADL recognition module")
    parser.add_argument("--pose-dir", default=str(OUTPUT_DIR / "3_pose"))
    parser.add_argument("--video-dir", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "4_adl"))
    parser.add_argument("--window-size", type=int, default=int(phase3_config.get("window_size", 30)))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.config:
        print(f"[WARN] --config is accepted for CLI compatibility; module uses local defaults: {args.config}")
    process_folder(args.pose_dir, args.video_dir, args.output, args.window_size)


if __name__ == "__main__":
    main()
