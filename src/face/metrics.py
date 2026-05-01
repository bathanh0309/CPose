from __future__ import annotations

from collections import Counter
from typing import Any


def build_face_metrics(events: list[dict[str, Any]], elapsed_sec: float, output_json_path: str) -> dict[str, Any]:
    total = len(events)
    detected = sum(1 for event in events if event.get("face_detected"))
    embedding_count = sum(1 for event in events if event.get("embedding") is not None)
    spoof_checked = sum(1 for event in events if event.get("spoof_status") not in {None, "unchecked"})
    unchecked = sum(1 for event in events if event.get("spoof_status") == "unchecked")
    failure_distribution = Counter(event.get("failure_reason", "OK") for event in events)
    return {
        "metric_type": "proxy",
        "total_face_events": total,
        "face_detected_count": detected,
        "face_detection_rate": detected / total if total else 0.0,
        "embedding_count": embedding_count,
        "spoof_checked_count": spoof_checked,
        "unchecked_count": unchecked,
        "failure_reason_distribution": dict(failure_distribution),
        "elapsed_sec": elapsed_sec,
        "fps_equivalent": total / elapsed_sec if elapsed_sec > 0 else 0.0,
        "output_json_path": output_json_path,
        "failure_reason": "OK",
    }
