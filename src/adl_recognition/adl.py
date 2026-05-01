"""Deprecated ADL module.

The single supported ADL engine is ``adl_rules.py``.
"""
from __future__ import annotations

from src.adl_recognition.adl_rules import classify_adl, history_item

__all__ = ["classify_adl", "history_item"]
