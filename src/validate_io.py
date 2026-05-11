from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_ADL = {"standing", "sitting", "walking", "lying_down", "falling", "reaching", "bending", "unknown"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_outputs(run_dir: Path) -> None:
    for path in run_dir.rglob("detections.json"):
        for frame in _load(path):
            assert "frame_id" in frame, f"Missing frame_id in {path}"
            for detection in frame.get("detections", []):
                assert detection.get("class_id") == 0, f"Non-person class in {path}"
                assert "failure_reason" in detection, f"Missing detection failure_reason in {path}"

    for path in run_dir.rglob("tracks.json"):
        for frame in _load(path):
            for track in frame.get("tracks", []):
                assert "global_id" not in track, f"global_id found before ReID in {path}"

    for path in run_dir.rglob("adl_events.json"):
        for event in _load(path):
            assert event.get("adl_label") in VALID_ADL, f"Invalid ADL label in {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CPose intermediate JSON outputs")
    parser.add_argument("--run-dir", default="dataset/outputs")
    args = parser.parse_args()
    validate_outputs(Path(args.run_dir))
    print("All IO schemas valid.")


if __name__ == "__main__":
    main()
