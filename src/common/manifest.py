from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoManifestItem:
    path: str
    camera_id: str | None = None
    start_time: str | None = None


def load_multicam_manifest(path: str | Path | None) -> list[VideoManifestItem]:
    if path is None or not Path(path).exists():
        print(f"[WARN] Manifest not found: {path}")
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [VideoManifestItem(**item) for item in payload.get("videos", [])]
