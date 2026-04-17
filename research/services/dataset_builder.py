# research/services/dataset_builder.py
from pathlib import Path
from typing import List

from shared.io.paths import get_data_dir

def list_raw_videos() -> List[Path]:
    d = get_data_dir() / "raw_videos"
    return sorted(p for p in d.glob("*.mp4"))

def list_pose_outputs() -> List[Path]:
    d = get_data_dir() / "output_pose"
    return sorted(p for p in d.glob("*.json"))
