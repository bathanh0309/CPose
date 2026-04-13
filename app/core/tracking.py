from app.utils.runtime_config import get_runtime_section

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as exc:  # pragma: no cover
    DeepSort = None
    _DEEPSORT_IMPORT_ERROR = exc

_TRACKER_DEFAULTS = get_runtime_section("tracker")

class Tracker:
    def __init__(self, embedder_name = 'mobilenet'):
        if DeepSort is None:  # pragma: no cover
            raise ImportError(
                "deep_sort_realtime is required for Tracker. Install it, then retry."
            ) from _DEEPSORT_IMPORT_ERROR
        self.model = DeepSort(
            max_age=int(_TRACKER_DEFAULTS.get("max_age", 30)),
            n_init=int(_TRACKER_DEFAULTS.get("n_init", 3)),
            max_iou_distance=float(_TRACKER_DEFAULTS.get("max_iou_distance", 0.7)),
            max_cosine_distance=float(_TRACKER_DEFAULTS.get("max_cosine_distance", 0.4)),
            embedder=embedder_name,
            half=bool(_TRACKER_DEFAULTS.get("half", True))
        )
        
    def update_tracks(self, detections, frame):
        return self.model.update_tracks(detections, frame=frame)
