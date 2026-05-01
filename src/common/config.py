from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.common.paths import resolve_path


DEFAULT_RESEARCH_CONFIG = Path(__file__).resolve().parent / "configs" / "config.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_research_config() -> dict[str, Any]:
    return load_yaml(DEFAULT_RESEARCH_CONFIG)


def get_research_section(section: str) -> dict[str, Any]:
    value = load_research_config().get(section, {})
    return dict(value) if isinstance(value, dict) else {}
