from __future__ import annotations

from pathlib import Path
from typing import Any


def load_camera_topology(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        print(f"[WARN] Camera topology not found: {path}")
        return {"cameras": {}, "transitions": []}
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {"cameras": {}, "transitions": []}
    except Exception as exc:
        print(f"[WARN] Could not load topology {path}: {exc}")
        return {"cameras": {}, "transitions": []}


def is_transition_allowed(topology: dict[str, Any], from_camera: str, to_camera: str, elapsed_sec: float) -> bool:
    for transition in topology.get("transitions", []):
        if transition.get("from") == from_camera and transition.get("to") == to_camera:
            return float(transition.get("min_sec", 0)) <= elapsed_sec <= float(transition.get("max_sec", 10**9))
    return False
