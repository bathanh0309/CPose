from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

from src.common.paths import MULTICAM_DIR, OUTPUT_DIR
from src.human_detection.api import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="CPose person detection module")
    parser.add_argument("--input", default=str(MULTICAM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "1_detection"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.config:
        print(f"[WARN] --config is accepted for CLI compatibility; module uses local defaults: {args.config}")
    process_folder(args.input, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()
