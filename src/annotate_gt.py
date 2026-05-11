from __future__ import annotations

import argparse
import json
from pathlib import Path


def bootstrap_detection_gt(video: Path, output_dir: Path, bbox: list[float] | None = None) -> Path:
    bbox = bbox or [100, 50, 250, 450]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": video.name,
        "frames": [
            {
                "frame_id": 0,
                "persons": [{"bbox": bbox, "person_id": 1, "visible": True}],
            }
        ],
    }
    output_path = output_dir / f"{video.stem}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a minimal detection GT JSON file")
    parser.add_argument("--video", required=True, help="Video file to annotate")
    parser.add_argument("--output-dir", default="dataset/annotations/gt-person-detection")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    output_path = bootstrap_detection_gt(Path(args.video), Path(args.output_dir), args.bbox)
    print(f"Sample GT created: {output_path}")


if __name__ == "__main__":
    main()
