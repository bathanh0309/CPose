"""Camera pipeline orchestration."""
from __future__ import annotations

from src.pipeline.cam1_pipeline import Cam1Pipeline
from src.pipeline.cam2_pipeline import Cam2Pipeline
from src.pipeline.slave_pipeline import SlavePipeline

__all__ = ["Cam1Pipeline", "Cam2Pipeline", "SlavePipeline"]
