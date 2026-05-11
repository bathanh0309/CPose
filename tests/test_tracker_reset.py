from __future__ import annotations

from models.tracking.tracker import SimpleIoUTracker, YoloByteTracker


def test_yolo_byte_tracker_reset_clears_cached_model_state(monkeypatch) -> None:
    class Model:
        predictor = object()

    model = Model()
    monkeypatch.setattr("src.modules.tracking.tracker._get_yolo_model", lambda _model_path: model)

    tracker = YoloByteTracker("dummy.pt")
    tracker.track_states[7] = {"hits": 4}
    tracker.frame_index = 12

    tracker.reset()

    assert tracker.track_states == {}
    assert tracker.frame_index == 0
    assert model.predictor is None


def test_simple_iou_tracker_uses_linear_motion_prediction() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.4, min_hits=1)

    first = tracker.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9}])
    second = tracker.update([{"bbox": [4, 0, 14, 10], "confidence": 0.9}])
    third = tracker.update([{"bbox": [8, 0, 18, 10], "confidence": 0.9}])

    assert first[0]["track_id"] == second[0]["track_id"] == third[0]["track_id"]
