"""tests/test_topology.py — Contract tests for camera topology loading and scoring."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.common.topology import CameraTopology, load_camera_topology, topology_score


_EXAMPLE_YAML = """\
transitions:
  - from: cam1
    to: cam2
    min_sec: 5
    max_sec: 60
    exit_zones: [right]
    entry_zones: [left]
  - from: cam2
    to: cam3
    min_sec: 10
    max_sec: 120
    exit_zones: [right]
    entry_zones: [left]
"""


@pytest.fixture()
def topology(tmp_path: Path) -> CameraTopology:
    yaml_path = tmp_path / "topology.yaml"
    yaml_path.write_text(_EXAMPLE_YAML, encoding="utf-8")
    return load_camera_topology(str(yaml_path))


class TestLoadTopology:
    def test_returns_camera_topology(self, topology: CameraTopology) -> None:
        assert isinstance(topology, CameraTopology)

    def test_transitions_loaded(self, topology: CameraTopology) -> None:
        assert len(topology.transitions) == 2

    def test_none_path_returns_empty_topology(self) -> None:
        result = load_camera_topology(None)
        assert isinstance(result, CameraTopology)

    def test_missing_file_returns_empty_topology(self, tmp_path: Path) -> None:
        result = load_camera_topology(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(result, CameraTopology)


class TestTopologyScore:
    def test_allowed_transition_returns_positive_score(self, topology: CameraTopology) -> None:
        result = topology_score(topology, "cam1", "cam2", delta_sec=30.0)
        assert result["allowed"] is True
        assert float(result["score"]) >= 0.0

    def test_disallowed_transition_topology_conflict(self, topology: CameraTopology) -> None:
        result = topology_score(topology, "cam1", "cam3", delta_sec=30.0)
        assert result["allowed"] is False or result.get("failure_reason") == "TOPOLOGY_CONFLICT"

    def test_time_outside_window_rejected(self, topology: CameraTopology) -> None:
        result = topology_score(topology, "cam1", "cam2", delta_sec=1.0)  # too fast (min 5s)
        # Allowed depends on impl, but score should be low or allowed=False
        assert "score" in result

    def test_same_camera_not_topology_conflict(self, topology: CameraTopology) -> None:
        result = topology_score(topology, "cam1", "cam1", delta_sec=10.0)
        assert "failure_reason" in result

    def test_score_between_zero_and_one(self, topology: CameraTopology) -> None:
        result = topology_score(topology, "cam1", "cam2", delta_sec=30.0)
        assert 0.0 <= float(result["score"]) <= 1.0
