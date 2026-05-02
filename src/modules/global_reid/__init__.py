"""CPose Module 5: global ReID."""
from __future__ import annotations

import sys

from src.modules.global_reid import matching as fusion_score
from src.modules.global_reid.api import process_clip, process_folder, run_global_reid

sys.modules.setdefault(__name__ + ".fusion_score", fusion_score)

__all__ = ["process_clip", "process_folder", "run_global_reid"]
