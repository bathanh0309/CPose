from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_dataset(data_dir: Path, manifest_path: Path | None) -> None:
    videos = sorted(data_dir.glob("*.mp4"))
    print(f"Videos found: {len(videos)}")
    for video in videos:
        print(f"  {video.name}")

    if manifest_path is None or not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_stems = {Path(entry["path"]).stem for entry in manifest.get("videos", [])}
    video_stems = {video.stem for video in videos}
    missing_from_manifest = video_stems - manifest_stems
    extra_in_manifest = manifest_stems - video_stems
    if missing_from_manifest:
        print(f"Videos not in manifest: {sorted(missing_from_manifest)}")
    if extra_in_manifest:
        print(f"Manifest entries without video file: {sorted(extra_in_manifest)}")
    if not missing_from_manifest and not extra_in_manifest:
        print("Manifest matches data directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CPose dataset and optional multicam manifest")
    parser.add_argument("--data-dir", default="data-test")
    parser.add_argument("--manifest", default="configs/camera/multicam_manifest.json")
    args = parser.parse_args()
    manifest = Path(args.manifest) if args.manifest else None
    check_dataset(Path(args.data_dir), manifest)


if __name__ == "__main__":
    main()
