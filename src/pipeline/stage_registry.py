"""Pipeline stage registry.

This module owns the canonical CPose stage order and the import targets used by
the orchestrator. Keep model logic inside module APIs, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    key: str
    module_name: str
    function_name: str = "process_folder"
    output_dir_name: str | None = None
    enabled: bool = True


DEFAULT_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("Detection", "detection", "src.modules.detection.api", output_dir_name="01_detection"),
    StageSpec("Tracking", "tracking", "src.modules.tracking.api", output_dir_name="02_tracking"),
    StageSpec("Pose", "pose", "src.modules.pose_estimation.api", output_dir_name="03_pose"),
    StageSpec("ADL", "adl", "src.modules.adl_recognition.api", output_dir_name="04_adl"),
    StageSpec("ReID", "reid", "src.modules.global_reid.api", output_dir_name="05_global_reid"),
)


def stage_output_dirs(run_dir: str | Path) -> dict[str, Path]:
    """Return canonical output directories for all registered pipeline stages."""

    root = Path(run_dir)
    return {stage.key: root / stage.output_dir_name for stage in DEFAULT_STAGE_SPECS if stage.output_dir_name}


def enabled_stages() -> tuple[StageSpec, ...]:
    """Return stages enabled for this run while preserving canonical order."""

    return tuple(stage for stage in DEFAULT_STAGE_SPECS if stage.enabled)


@dataclass(frozen=True, slots=True)
class StageCall:
    spec: StageSpec
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


__all__ = ["DEFAULT_STAGE_SPECS", "StageCall", "StageSpec", "enabled_stages", "stage_output_dirs"]
