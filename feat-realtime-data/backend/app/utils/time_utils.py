# Hàm tiện ích chuẩn hóa timestamp UTC và parse chuỗi ISO.
"""Shared time utilities."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 string (with or without Z) to a UTC datetime."""
    ts = ts.rstrip("Z")
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
