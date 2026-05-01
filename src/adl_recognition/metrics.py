from __future__ import annotations

from collections import Counter


def build_adl_metrics(events: list[dict], elapsed_sec: float, output_json_path: str, output_video_path: str | None) -> dict:
    distribution = Counter(event["adl_label"] for event in events)
    raw_distribution = Counter(event.get("raw_label", event["adl_label"]) for event in events)
    failure_distribution = Counter(event.get("failure_reason", "OK") for event in events)
    total = len(events)
    smoothing_changed = sum(1 for event in events if event.get("raw_label") != event.get("smoothed_label"))
    return {
        "total_adl_events": total,
        "class_distribution": dict(distribution),
        "raw_class_distribution": dict(raw_distribution),
        "unknown_rate": distribution.get("unknown", 0) / total if total else 0.0,
        "avg_confidence": (
            sum(float(event["confidence"]) for event in events if event.get("confidence") is not None)
            / sum(1 for event in events if event.get("confidence") is not None)
            if any(event.get("confidence") is not None for event in events)
            else None
        ),
        "avg_window_latency_ms": (elapsed_sec / total * 1000.0) if total else 0.0,
        "fps_equivalent": total / elapsed_sec if elapsed_sec > 0 else 0.0,
        "elapsed_sec": elapsed_sec,
        "smoothing_changed_count": smoothing_changed,
        "failure_reason_distribution": dict(failure_distribution),
        "output_json_path": output_json_path,
        "output_video_path": output_video_path,
    }
