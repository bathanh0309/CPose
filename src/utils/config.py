from copy import deepcopy
from pathlib import Path

import yaml


PATH_FIELDS = {
    ("system", "event_log"),
    ("system", "vis_dir"),
    ("system", "default_source"),
    ("pose", "weights"),
    ("reid", "fastreid_root"),
    ("reid", "config"),
    ("reid", "weights"),
    ("reid", "gallery_dir"),
    ("adl", "mmaction_root"),
    ("adl", "base_config"),
    ("adl", "weights"),
    ("adl", "export_dir"),
    ("adl", "work_dir"),
}

ULTRALYTICS_BUILTIN_TRACKERS = {"bytetrack.yaml", "botsort.yaml"}


def normalize_device(value):
    if value is None:
        return None

    device = str(value).strip().lower()
    if device in {"", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    if device.startswith("cuda") or device.isdigit():
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"

    return device


def resolve_project_path(value, root: Path) -> str:
    """Resolve a config path relative to the CPose project root."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def resolve_tracker_yaml(value, root: Path) -> str:
    tracker_yaml = str(value).strip()
    if tracker_yaml in ULTRALYTICS_BUILTIN_TRACKERS:
        return tracker_yaml

    path = Path(tracker_yaml).expanduser()
    if path.is_absolute():
        return str(path)

    # Project-relative paths contain a path separator. Bare names are left to
    # Ultralytics, e.g. bytetrack.yaml and botsort.yaml.
    if any(sep in tracker_yaml for sep in ("/", "\\")):
        return str((root / path).resolve())

    candidate = (root / path).resolve()
    return str(candidate) if candidate.exists() else tracker_yaml


def resolve_cfg_paths(cfg: dict, root: Path) -> dict:
    root = Path(root).resolve()
    cfg = deepcopy(cfg)

    for section, key in PATH_FIELDS:
        value = cfg.get(section, {}).get(key)
        if not value:
            continue
        cfg[section][key] = resolve_project_path(value, root)

    if "system" in cfg:
        cfg["system"]["device"] = normalize_device(cfg["system"].get("device"))

    tracker_yaml = cfg.get("tracker", {}).get("tracker_yaml")
    if tracker_yaml:
        cfg["tracker"]["tracker_yaml"] = resolve_tracker_yaml(tracker_yaml, root)

    embedding_dirs = cfg.get("reid", {}).get("embedding_dirs")
    if embedding_dirs:
        cfg["reid"]["embedding_dirs"] = [
            resolve_project_path(path, root) for path in embedding_dirs
        ]

    return cfg


def load_pipeline_cfg(path: Path, root: Path) -> dict:
    root = Path(root).resolve()
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validate_cfg(cfg)
    return resolve_cfg_paths(cfg, root)


def validate_cfg(cfg: dict):
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")

    required = {
        "pose": ["weights", "conf", "iou"],
        "reid": ["config", "weights", "gallery_dir", "threshold", "fastreid_root", "reid_interval"],
        "adl": ["mmaction_root", "base_config", "weights", "seq_len", "stride", "export_dir"],
        "system": ["device", "event_log", "vis_dir"],
        "tracker": ["tracker_yaml"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"Missing config section: [{section}]")
        section_cfg = cfg.get(section, {})
        for key in keys:
            if key not in section_cfg:
                raise ValueError(f"Missing config: [{section}].{key}")
