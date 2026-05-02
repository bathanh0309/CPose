"""Temporal smoothing helpers for ADL labels."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def majority_vote(labels: Iterable[str], default: str = "unknown") -> str:
    counts = Counter(labels)
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0] == "unknown", item[0]))[0][0]


__all__ = ["majority_vote"]
