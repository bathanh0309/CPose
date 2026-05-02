from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from src.common.errors import ErrorCode, as_reason
from src.common.paths import PROJECT_ROOT, ensure_dir, resolve_path
from src.common.persistence import PersistenceManager
from src.common.schemas import FrameRecord, to_dict
from src.common.video_io import list_video_files
from src.common.visualization import draw_bbox, draw_global_id, draw_skeleton


def test_resolve_path_anchors_relative_paths_to_project_root(tmp_path: Path) -> None:
    assert resolve_path("dataset") == PROJECT_ROOT / "dataset"
    assert resolve_path(tmp_path) == tmp_path

    created = ensure_dir(tmp_path / "nested" / "folder")

    assert created.exists()
    assert created.is_dir()


def test_schema_to_dict_serializes_common_types() -> None:
    payload = {
        "frame": FrameRecord(frame_id=1, timestamp_sec=0.5, camera_id="cam1"),
        "path": Path("dataset/runs/demo"),
        "created_at": datetime(2026, 5, 1, 12, 30, 0),
    }

    result = to_dict(payload)

    assert result["frame"]["failure_reason"] == "OK"
    assert result["path"].replace("\\", "/") == "dataset/runs/demo"
    assert result["created_at"] == "2026-05-01T12:30:00"


def test_as_reason_preserves_known_codes_and_falls_back_for_unknown() -> None:
    assert as_reason(ErrorCode.OK) == "OK"
    assert as_reason("NO_FACE") == "NO_FACE"
    assert as_reason("unexpected") == "STEP_FAILED"


def test_list_video_files_orders_by_earliest_filename_timestamp(tmp_path: Path) -> None:
    filenames = [
        "cam2_2026-01-29_16-26-40.mp4",
        "cam1_2026-01-29_16-26-25.mp4",
        "cam4_2026-01-28_15-59-10.mp4",
    ]
    for filename in filenames:
        (tmp_path / filename).write_bytes(b"")

    result = [path.name for path in list_video_files(tmp_path)]

    assert result == [
        "cam4_2026-01-28_15-59-10.mp4",
        "cam1_2026-01-29_16-26-25.mp4",
        "cam2_2026-01-29_16-26-40.mp4",
    ]


def test_visualization_draw_helpers_modify_frame() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    draw_bbox(frame, [10, 10, 50, 70], "person")
    draw_skeleton(
        frame,
        [
            {"id": 5, "x": 30, "y": 30, "confidence": 0.9},
            {"id": 7, "x": 35, "y": 55, "confidence": 0.9},
        ],
    )
    draw_global_id(frame, [60, 20, 110, 90], 1, adl_label="standing")

    assert int(frame.sum()) > 0


def test_draw_skeleton_ignores_invalid_origin_keypoints() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    draw_skeleton(
        frame,
        [
            {"id": 5, "x": 0, "y": 0, "confidence": 0.99},
            {"id": 7, "x": 80, "y": 80, "confidence": 0.99},
        ],
    )

    assert int(frame[:8, :8].sum()) == 0


def test_persistence_manager_round_trip(tmp_path: Path) -> None:
    manager = PersistenceManager(str(tmp_path / "state"), embedding_dim=4)
    try:
        global_id = manager.get_next_global_id()
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        manager.register_global_id(global_id, "cam1", embedding, bbox=[1, 2, 3, 4])

        np.testing.assert_allclose(manager.get_embedding(global_id), embedding)
        assert manager.get_last_seen(global_id, "cam1") is not None
        assert manager.get_statistics()["total_global_ids"] == 1
    finally:
        manager.close()
