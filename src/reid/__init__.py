"""Identity matching and gallery management."""
from __future__ import annotations

from src.reid.gallery import FaceGallery
from src.reid.global_id_manager import GlobalIDManager, GlobalPerson, GlobalPersonTable
from src.reid.matcher import MultiCameraMatcher

__all__ = ["FaceGallery", "GlobalIDManager", "GlobalPerson", "GlobalPersonTable", "MultiCameraMatcher"]
