from __future__ import annotations

import argparse

from src.face.api import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional CPose face module")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track-dir", default=None)
    parser.add_argument("--run-every-n-frames", type=int, default=10)
    args = parser.parse_args()
    process_folder(args.input, args.output, args.track_dir, run_every_n_frames=args.run_every_n_frames)


if __name__ == "__main__":
    main()
