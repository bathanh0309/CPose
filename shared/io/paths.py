import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.absolute()

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "configs"

# Sub-dirs
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
OUTPUT_POSE_DIR = DATA_DIR / "output_pose"
MULTICAM_DIR = DATA_DIR / "multicam"
RESEARCH_RUNS_DIR = DATA_DIR / "research_runs"

def ensure_dirs():
    """Create all standard directories if they don't exist"""
    for d in [DATA_DIR, MODELS_DIR, CONFIGS_DIR, RAW_VIDEOS_DIR, OUTPUT_POSE_DIR, MULTICAM_DIR, RESEARCH_RUNS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
