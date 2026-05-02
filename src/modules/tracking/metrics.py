"""Tracking proxy metric builders."""
from __future__ import annotations


def build_tracking_metrics(records: list[dict], elapsed_sec: float, output_video_path: str, output_json_path: str) -> dict:
    processed_frames = len(records)
    track_ids: set[int] = set()
    active_counts: list[int] = []
    confirmed = 0
    unconfirmed = 0
    quality_values: list[float] = []
    first_seen: dict[int, int] = {}
    fragments = 0
    previous_active: set[int] = set()
    for record in records:
        current_active = {int(track["track_id"]) for track in record["tracks"]}
        active_counts.append(len(current_active))
        for track_id in current_active:
            track_ids.add(track_id)
            if track_id in first_seen and track_id not in previous_active:
                fragments += 1
            first_seen.setdefault(track_id, record["frame_id"])
        for track in record["tracks"]:
            confirmed += 1 if track.get("is_confirmed") else 0
            unconfirmed += 0 if track.get("is_confirmed") else 1
            if track.get("quality_score") is not None:
                quality_values.append(float(track["quality_score"]))
        previous_active = current_active
    return {
        "total_frames": processed_frames,
        "total_tracks": len(track_ids),
        "active_track_count_mean": sum(active_counts) / processed_frames if processed_frames else 0.0,
        "proxy_track_fragmentation": fragments,
        "track_fragmentation_proxy": fragments,
        "confirmed_track_count": confirmed,
        "unconfirmed_track_count": unconfirmed,
        "avg_track_quality": sum(quality_values) / len(quality_values) if quality_values else None,
        "id_switch_proxy": None,
        "fps_processing": processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
        "avg_latency_ms_per_frame": (elapsed_sec / processed_frames * 1000.0) if processed_frames else 0.0,
        "elapsed_sec": elapsed_sec,
        "output_video_path": output_video_path,
        "output_json_path": output_json_path,
    }


__all__ = ["build_tracking_metrics"]
