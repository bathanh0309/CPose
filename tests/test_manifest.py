"""tests/test_manifest.py — Contract tests for manifest loading and resolution."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.manifest import resolve_videos_from_manifest, ResolvedVideoItem


class TestResolveVideosNoManifest:
    """Fallback: no manifest file — should glob videos from the directory."""

    def test_returns_list(self, tmp_path: Path) -> None:
        # Create two stub video files
        (tmp_path / "cam1.mp4").write_bytes(b"")
        (tmp_path / "cam2.mp4").write_bytes(b"")
        result = resolve_videos_from_manifest(tmp_path, None)
        assert isinstance(result, list)

    def test_no_videos_returns_empty(self, tmp_path: Path) -> None:
        result = resolve_videos_from_manifest(tmp_path, None)
        assert result == []

    def test_items_have_required_fields(self, tmp_path: Path) -> None:
        (tmp_path / "cam1.mp4").write_bytes(b"")
        result = resolve_videos_from_manifest(tmp_path, None)
        assert len(result) >= 1
        item = result[0]
        assert hasattr(item, "path")
        assert hasattr(item, "camera_id")
        assert isinstance(item.path, Path)

    def test_orders_by_earliest_filename_timestamp(self, tmp_path: Path) -> None:
        for filename in [
            "cam2_2026-01-29_16-26-40.mp4",
            "cam1_2026-01-29_16-26-25.mp4",
            "cam4_2026-01-28_15-59-10.mp4",
        ]:
            (tmp_path / filename).write_bytes(b"")

        result = resolve_videos_from_manifest(tmp_path, None)

        assert [item.video for item in result] == [
            "cam4_2026-01-28_15-59-10.mp4",
            "cam1_2026-01-29_16-26-25.mp4",
            "cam2_2026-01-29_16-26-40.mp4",
        ]


class TestResolveVideosWithManifest:
    """With a manifest JSON — timestamps and camera IDs must be honoured."""

    def _make_manifest(self, tmp_path: Path, videos: list[dict]) -> Path:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"videos": videos}), encoding="utf-8")
        return manifest_path

    def test_camera_id_from_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "v.mp4").write_bytes(b"")
        manifest = self._make_manifest(tmp_path, [{"path": str(tmp_path / "v.mp4"), "camera_id": "camA", "start_time": "2026-01-01T00:00:00+07:00"}])
        result = resolve_videos_from_manifest(tmp_path, manifest)
        assert any(item.camera_id == "camA" for item in result)

    def test_start_time_parsed(self, tmp_path: Path) -> None:
        (tmp_path / "v.mp4").write_bytes(b"")
        manifest = self._make_manifest(tmp_path, [{"path": str(tmp_path / "v.mp4"), "camera_id": "cam1", "start_time": "2026-01-29T16:26:25+07:00"}])
        result = resolve_videos_from_manifest(tmp_path, manifest)
        items = [i for i in result if i.camera_id == "cam1"]
        assert items
        assert items[0].start_time is not None

    def test_missing_video_file_handled_gracefully(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, [{"path": str(tmp_path / "missing.mp4"), "camera_id": "cam1", "start_time": None}])
        # Should not raise — missing files are handled gracefully (items may be empty or missing)
        result = resolve_videos_from_manifest(tmp_path, manifest)
        assert isinstance(result, list)

    def test_malformed_manifest_returns_empty_or_fallback(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "bad.json"
        manifest_path.write_text("{invalid json", encoding="utf-8")
        result = resolve_videos_from_manifest(tmp_path, manifest_path)
        assert isinstance(result, list)
