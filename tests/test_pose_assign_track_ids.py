from __future__ import annotations

from models.pose_estimation.api import _assign_track_ids


def test_assign_track_ids_returns_new_person_dicts_without_mutating_input() -> None:
    persons = [{"bbox": [0, 0, 10, 10], "failure_reason": "OK"}]
    tracks = [{"track_id": 5, "bbox": [0, 0, 10, 10], "is_confirmed": True, "failure_reason": "OK"}]

    result = _assign_track_ids(persons, tracks)

    assert result[0]["track_id"] == 5
    assert "track_id" not in persons[0]
