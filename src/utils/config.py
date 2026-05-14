from pathlib import Path

import yaml


PATH_FIELDS = {
    ("system", "event_log"),
    ("system", "vis_dir"),
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


def resolve_cfg_paths(cfg: dict, root: Path) -> dict:
    for section, key in PATH_FIELDS:
        value = cfg.get(section, {}).get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            cfg[section][key] = str(root / path)
    return cfg


def load_pipeline_cfg(path: Path, root: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_cfg(cfg)
    return resolve_cfg_paths(cfg, root)


def validate_cfg(cfg: dict):
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")

    required = {
        "pose": ["weights", "conf", "iou"],
        "reid": ["config", "weights", "gallery_dir", "threshold", "fastreid_root", "reid_interval"],
        "adl": ["seq_len", "stride", "max_idle_frames", "export_dir"],
        "system": ["device", "event_log", "vis_dir"],
        "tracker": ["tracker_yaml"],
    }
    for section, keys in required.items():
        section_cfg = cfg.get(section, {})
        for key in keys:
            if key not in section_cfg:
                raise ValueError(f"Missing config: [{section}].{key}")
