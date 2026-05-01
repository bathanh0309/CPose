"""Load runtime defaults from app/config.yaml."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("[RuntimeConfig]")

_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_FILE = _BASE_DIR / "app" / "config.yaml"
_CONFIG_ENV_VAR = "CPOSE_RUNTIME_CONFIG"


def _resolve_config_file() -> Path:
    raw_path = os.environ.get(_CONFIG_ENV_VAR, "").strip()
    if not raw_path:
        return _DEFAULT_CONFIG_FILE
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = _BASE_DIR / candidate
    return candidate


@lru_cache(maxsize=1)
def load_runtime_config() -> dict[str, Any]:
    config_file = _resolve_config_file()
    if not config_file.exists():
        logger.warning("Runtime config file not found: %s", config_file)
        return {}
    try:
        with config_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load runtime config %s: %s", config_file, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Runtime config root must be a dict: %s", config_file)
        return {}
    return data


def get_runtime_section(section: str) -> dict[str, Any]:
    section_data = load_runtime_config().get(section, {})
    if not isinstance(section_data, dict):
        return {}
    return dict(section_data)
