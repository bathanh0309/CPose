from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.common.errors import ErrorCode
from src.paths import (
    ANNOTATIONS_DIR,
    DATA_DIR,
    MODELS_DIR,
    MULTICAM_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_dir,
    resolve_path,
)


RTSP_CAM1 = os.getenv("RTSP_CAM1", "")
RTSP_CAM2 = os.getenv("RTSP_CAM2", "")
FACE_GALLERY_DIR = PROJECT_ROOT / "data" / "face"
LIVENESS_MODEL = MODELS_DIR / "face_antispoof" / "best_model_quantized.onnx"
PERSON_DETECTOR_MODEL = MODELS_DIR / "human_detect" / "yolov8n.pt"
POSE_MODEL = MODELS_DIR / "pose_estimation" / "yolov8n-pose.pt"
HOMOGRAPHY_PATH = MODELS_DIR / "camera" / "homography.npy"


CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG_DIR = CONFIG_DIR / "base"
CAMERA_CONFIG_DIR = CONFIG_DIR / "camera"
PROFILE_CONFIG_DIR = CONFIG_DIR / "profiles"
DEFAULT_PROFILE = os.getenv("CPOSE_PROFILE", "dev")
DEFAULT_MODEL_REGISTRY = PROFILE_CONFIG_DIR / f"{DEFAULT_PROFILE}.yaml"
DEFAULT_RESEARCH_CONFIG = CONFIG_DIR / "unified_config.yaml"
BASE_CONFIG_LOAD_ORDER = (
    "models.yaml",
    "thresholds.yaml",
    "pipeline.yaml",
    "adl.yaml",
    "logging.yaml",
)

SECTION_ALIASES = {
    "human_detection": "detector",
    "human_tracking": "tracker",
    "pose_estimation": "pose",
    "adl_recognition": "adl",
    "global_reid": "reid",
}


@dataclass(slots=True)
class ResolvedModel:
    section: str
    name: str | None
    path: Path | None
    requested_path: str | None
    fallback_used: bool
    fallback_name: str | None
    conf: float | None
    params: dict[str, Any]
    failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path) if self.path is not None else None
        return payload


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _profile_name(profile: str | Path | None) -> str:
    if profile is None:
        return DEFAULT_PROFILE
    profile_path = Path(str(profile))
    return profile_path.stem if profile_path.suffix in {".yaml", ".yml"} else str(profile)


def load_config(profile: str | Path | None = None) -> dict[str, Any]:
    """Load base config, camera metadata, phase, then apply a profile override."""

    cfg: dict[str, Any] = {}
    for filename in BASE_CONFIG_LOAD_ORDER:
        cfg = _deep_merge(cfg, load_yaml(BASE_CONFIG_DIR / filename))

    topology = load_yaml(CAMERA_CONFIG_DIR / "topology.yaml")
    if topology:
        cfg["cameras"] = topology
        cfg["camera_topology"] = topology

    manifest = _load_json(CAMERA_CONFIG_DIR / "multicam_manifest.json")
    if manifest:
        cfg["multicam_manifest"] = manifest

    profile_name = _profile_name(profile)
    profile_file = PROFILE_CONFIG_DIR / f"{profile_name}.yaml"
    if profile_file.exists():
        cfg = _deep_merge(cfg, load_yaml(profile_file))
    elif profile_name and profile_name != "base":
        print(f"[WARN] Config profile not found: {profile_file}")

    phase = load_yaml(CONFIG_DIR / "phase.yaml")
    if phase:
        cfg["phase"] = phase

    return cfg


@lru_cache(maxsize=1)
def load_research_config() -> dict[str, Any]:
    return load_yaml(DEFAULT_RESEARCH_CONFIG)


def get_research_section(section: str) -> dict[str, Any]:
    value = load_research_config().get(section, {})
    return dict(value) if isinstance(value, dict) else {}


def load_model_registry(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return load_config(DEFAULT_PROFILE)
    registry_path = resolve_path(path)
    if registry_path.parent == PROFILE_CONFIG_DIR:
        return load_config(registry_path.stem)
    if not registry_path.exists():
        profile_file = PROFILE_CONFIG_DIR / f"{_profile_name(path)}.yaml"
        if profile_file.exists():
            return load_config(profile_file.stem)
        print(f"[WARN] Model registry not found: {registry_path}")
        return {}
    try:
        return load_yaml(registry_path)
    except Exception as exc:
        print(f"[WARN] Could not load model registry {registry_path}: {exc}")
        return {}


def get_section(registry: dict[str, Any], section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = registry.get(section)
    if value is None:
        value = registry.get(SECTION_ALIASES.get(section, ""), default or {})
    return value if isinstance(value, dict) else {}


def _candidate_to_path(candidate: str | Path | None) -> Path | None:
    if not candidate:
        return None
    path = Path(str(candidate))
    if path.is_absolute():
        return path
    resolved = resolve_path(path)
    return resolved if resolved.exists() else path


def resolve_model_path(registry: dict[str, Any], section: str) -> ResolvedModel:
    item = get_section(registry, section)
    requested = item.get("model") or item.get("path")
    candidates = [requested, *(item.get("fallback") or [])]
    for index, candidate in enumerate(candidates):
        if not candidate:
            continue
        path = _candidate_to_path(candidate)
        if path is None:
            continue
        resolved_abs = resolve_path(path) if not path.is_absolute() else path
        if resolved_abs.exists() or Path(str(candidate)).name == str(candidate):
            return ResolvedModel(
                section=section,
                name=str(candidate),
                path=resolved_abs if resolved_abs.exists() else Path(str(candidate)),
                requested_path=str(requested) if requested else None,
                fallback_used=index > 0,
                fallback_name=str(candidate) if index > 0 else None,
                conf=float(item["conf"]) if item.get("conf") is not None else None,
                params={k: v for k, v in item.items() if k not in {"model", "path", "fallback"}},
                failure_reason=ErrorCode.OK.value,
            )
    print(f"[WARN] No model path found for registry section: {section}")
    return ResolvedModel(
        section=section,
        name=None,
        path=None,
        requested_path=str(requested) if requested else None,
        fallback_used=False,
        fallback_name=None,
        conf=float(item["conf"]) if item.get("conf") is not None else None,
        params={k: v for k, v in item.items() if k not in {"model", "path", "fallback"}},
        failure_reason=ErrorCode.MODEL_NOT_FOUND.value,
    )


def build_effective_runtime_config(registry: dict[str, Any], manifest: Any, topology: Any, args: Any) -> dict[str, Any]:
    sections = [
        "human_detection",
        "human_tracking",
        "pose_estimation",
        "adl_recognition",
        "face",
        "global_reid",
    ]
    return {
        "runtime": get_section(registry, "runtime"),
        "models": {section: resolve_model_path(registry, section).to_dict() for section in sections},
        "sections": {section: get_section(registry, section) for section in sections},
        "manifest_video_count": len(manifest) if manifest is not None else 0,
        "topology_transition_count": len(getattr(topology, "transitions", []) or []),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} if hasattr(args, "__dict__") else {},
    }
