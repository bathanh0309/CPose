from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.paths import resolve_path


def load_model_registry(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        print(f"[WARN] Model registry not found: {path}")
        return {}
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        print(f"[WARN] Could not load model registry {path}: {exc}")
        return {}


def resolve_model_path(registry: dict[str, Any], section: str) -> Path | None:
    item = registry.get(section, {})
    candidates = [item.get("path"), *item.get("fallback", [])]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve_path(candidate)
        if path.exists():
            return path
    print(f"[WARN] No model path found for registry section: {section}")
    return None
