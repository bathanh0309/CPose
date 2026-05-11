from __future__ import annotations

from src.cam2_pipeline import Cam2Pipeline


class SlavePipeline(Cam2Pipeline):
    """Camera-B slave pipeline placeholder.

    It currently reuses Cam2Pipeline detection/tracking and leaves identity
    authority to the master/global matcher layer.
    """


__all__ = ["SlavePipeline"]
