from __future__ import annotations

ACTIVE = "ACTIVE"
PENDING_TRANSFER = "PENDING_TRANSFER"
IN_BLIND_ZONE = "IN_BLIND_ZONE"
IN_ROOM = "IN_ROOM"
CLOTHING_CHANGE_SUSPECTED = "CLOTHING_CHANGE_SUSPECTED"
DORMANT = "DORMANT"
CLOSED = "CLOSED"


def next_missing_state(current: str, missing_sec: float, max_candidate_age_sec: float) -> str:
    if current == CLOSED:
        return CLOSED
    if missing_sec > max_candidate_age_sec:
        return DORMANT
    if current == ACTIVE:
        return PENDING_TRANSFER
    return current
