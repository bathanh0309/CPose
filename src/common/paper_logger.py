"""Helpers for writing paper-report artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.json_io import save_csv, save_json
from src.common.paths import ensure_dir, resolve_path


def paper_report_dir(run_dir: str | Path) -> Path:
    """Return and create the canonical paper report directory for a run."""

    return ensure_dir(resolve_path(run_dir) / "08_paper_report")


def write_report_text(run_dir: str | Path, filename: str, text: str) -> Path:
    """Write a UTF-8 text artifact under ``08_paper_report``."""

    path = paper_report_dir(run_dir) / filename
    path.write_text(text, encoding="utf-8")
    return path


def write_report_json(run_dir: str | Path, filename: str, payload: Any) -> Path:
    """Write a JSON artifact under ``08_paper_report``."""

    return save_json(paper_report_dir(run_dir) / filename, payload)


def write_report_csv(run_dir: str | Path, filename: str, rows: list[dict[str, Any]]) -> Path:
    """Write a CSV artifact under ``08_paper_report``."""

    return save_csv(paper_report_dir(run_dir) / filename, rows)


__all__ = ["paper_report_dir", "write_report_csv", "write_report_json", "write_report_text"]
