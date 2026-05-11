from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from models.adl_recognition.api import process_video


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    try:
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def _standing_person(x_offset: int) -> dict:
    keypoints = [
        {"id": 5, "x": 10 + x_offset, "y": 10, "confidence": 0.9},
        {"id": 6, "x": 14 + x_offset, "y": 10, "confidence": 0.9},
        {"id": 11, "x": 10 + x_offset, "y": 30, "confidence": 0.9},
        {"id": 12, "x": 14 + x_offset, "y": 30, "confidence": 0.9},
        {"id": 13, "x": 10 + x_offset, "y": 42, "confidence": 0.9},
        {"id": 14, "x": 14 + x_offset, "y": 42, "confidence": 0.9},
        {"id": 15, "x": 10 + x_offset, "y": 55, "confidence": 0.9},
        {"id": 16, "x": 14 + x_offset, "y": 55, "confidence": 0.9},
    ]
    return {
        "track_id": None,
        "is_confirmed": True,
        "visible_keypoint_count": 8,
        "visible_keypoint_ratio": 1.0,
        "bbox": [5 + x_offset, 5, 20 + x_offset, 60],
        "keypoints": keypoints,
        "frame_height": 64,
    }


def test_unmatched_persons_get_distinct_negative_track_ids(tmp_path: Path) -> None:
    video_path = tmp_path / "cam1_test.mp4"
    pose_dir = tmp_path / "pose"
    output_dir = tmp_path / "adl"
    _write_video(video_path)

    pose_clip_dir = pose_dir / video_path.stem
    pose_clip_dir.mkdir(parents=True)
    (pose_clip_dir / "keypoints.json").write_text(
        json.dumps(
            [
                {
                    "frame_id": 0,
                    "timestamp_sec": 0.0,
                    "camera_id": "cam1",
                    "persons": [_standing_person(0), _standing_person(30)],
                }
            ]
        ),
        encoding="utf-8",
    )

    process_video(video_path, output_dir, pose_dir, preview=False)

    events = json.loads((output_dir / video_path.stem / "adl_events.json").read_text(encoding="utf-8"))
    assert [event["track_id"] for event in events] == [-1, -2]
