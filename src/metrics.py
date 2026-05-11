"""Tracking proxy metric builders — paper-ready, no GT required."""
from __future__ import annotations

import math


def build_tracking_metrics(
    records: list[dict],
    elapsed_sec: float,
    output_video_path: str,
    output_json_path: str,
    trajectory_video_path: str | None = None,
    trajectory_snapshot_path: str | None = None,
    tracklets_json_path: str | None = None,
) -> dict:
    processed_frames = len(records)
    track_ids: set[int] = set()
    active_counts: list[int] = []
    confirmed_ids: set[int] = set()
    quality_values: list[float] = []
    speed_values: list[float] = []
    age_values: list[float] = []
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
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
            last_seen[track_id] = record["frame_id"]
        for track in record["tracks"]:
            if track.get("is_confirmed"):
                confirmed_ids.add(int(track["track_id"]))
            if track.get("quality_score") is not None:
                quality_values.append(float(track["quality_score"]))
            if track.get("speed_px_per_sec") is not None:
                speed_values.append(float(track["speed_px_per_sec"]))
            if track.get("age") is not None:
                age_values.append(float(track["age"]))
        previous_active = current_active

    total_tracks = len(track_ids)
    confirmed = len(confirmed_ids)
    confirmed_ratio = confirmed / total_tracks if total_tracks > 0 else 0.0
    mean_age = sum(age_values) / len(age_values) if age_values else 0.0
    fragment_proxy = fragments / max(total_tracks, 1) if total_tracks > 0 else 0.0
    fps_proc = processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0
    latency_ms = (elapsed_sec / processed_frames * 1000.0) if processed_frames > 0 else 0.0

    return {
        "total_frames": processed_frames,
        "total_tracks": total_tracks,
        "confirmed_tracks": confirmed,
        "confirmed_track_ratio": round(confirmed_ratio, 4),
        "active_track_count_mean": round(sum(active_counts) / processed_frames, 3) if processed_frames else 0.0,
        "mean_track_age": round(mean_age, 2),
        "mean_track_quality": round(sum(quality_values) / len(quality_values), 4) if quality_values else None,
        "mean_speed_px_per_sec": round(sum(speed_values) / len(speed_values), 2) if speed_values else None,
        "fragment_proxy": round(fragment_proxy, 4),
        "proxy_track_fragmentation": fragments,
        "track_fragmentation_proxy": fragments,
        "confirmed_track_count": confirmed,
        "unconfirmed_track_count": total_tracks - confirmed,
        "avg_track_quality": round(sum(quality_values) / len(quality_values), 4) if quality_values else None,
        "id_switch_proxy": None,
        "fps_processing": round(fps_proc, 2),
        "avg_latency_ms_per_frame": round(latency_ms, 2),
        "elapsed_sec": round(elapsed_sec, 3),
        "output_video_path": output_video_path,
        "output_json_path": output_json_path,
        "trajectory_video_path": trajectory_video_path,
        "trajectory_snapshot_path": trajectory_snapshot_path,
        "tracklets_json_path": tracklets_json_path,
    }


def build_tracklets(
    records: list[dict],
    fps: float,
    camera_id: str,
    track_history: dict,
    topology: dict | None = None,
) -> list[dict]:
    """Build per-track tracklet summary from per-frame records."""
    track_data: dict[int, dict] = {}

    for record in records:
        fid = int(record["frame_id"])
        for track in record.get("tracks", []):
            t_id = int(track["track_id"])
            if t_id not in track_data:
                track_data[t_id] = {
                    "start_frame": fid,
                    "end_frame": fid,
                    "observations": [],
                    "confidences": [],
                    "areas": [],
                    "aspects": [],
                    "trajectory": [],
                }
            td = track_data[t_id]
            td["end_frame"] = fid
            bbox = track.get("bbox", [0, 0, 0, 0])
            bc = track.get("bottom_center", [(bbox[0] + bbox[2]) / 2, bbox[3]])
            conf = track.get("confidence", 0.0)
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            area = w * h
            aspect = (w / h) if h > 0 else 0.0
            td["observations"].append(fid)
            td["confidences"].append(float(conf))
            td["areas"].append(area)
            td["aspects"].append(aspect)
            td["trajectory"].append({"frame_id": fid, "point": [round(bc[0], 1), round(bc[1], 1)]})

    tracklets = []
    fps = fps if fps > 0 else 25.0
    for t_id, td in track_data.items():
        n = len(td["observations"])
        start_f = td["start_frame"]
        end_f = td["end_frame"]
        dur = (end_f - start_f) / fps
        mean_conf = sum(td["confidences"]) / n if n > 0 else 0.0
        mean_area = sum(td["areas"]) / n if n > 0 else 0.0
        mean_aspect = sum(td["aspects"]) / n if n > 0 else 0.0
        traj = td["trajectory"]
        entry = traj[0]["point"] if traj else [0, 0]
        exit_pt = traj[-1]["point"] if traj else [0, 0]
        
        # Compute mean speed from history
        hist = track_history.get(t_id, [])
        speeds = []
        if len(hist) > 1:
            for i in range(1, len(hist)):
                dx = hist[i][0] - hist[i-1][0]
                dy = hist[i][1] - hist[i-1][1]
                speeds.append(math.sqrt(dx*dx + dy*dy) * fps)
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        
        quality = 0.5 * mean_conf + 0.3 * min(n / max(fps * 2, 1), 1.0) + 0.2 * min(dur / 5.0, 1.0)

        tracklets.append({
            "camera_id": camera_id,
            "local_track_id": f"T{t_id:03d}",
            "track_id_int": t_id,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_time_sec": round(start_f / fps, 3),
            "end_time_sec": round(end_f / fps, 3),
            "duration_sec": round(dur, 3),
            "num_observations": n,
            "mean_confidence": round(mean_conf, 4),
            "mean_bbox_area": round(mean_area, 1),
            "mean_aspect_ratio": round(mean_aspect, 3),
            "mean_speed_px_per_sec": round(mean_speed, 2),
            "entry_point": entry,
            "exit_point": exit_pt,
            "entry_zone": "unknown",
            "exit_zone": "unknown",
            "representative_crop": None,
            "trajectory": traj,
            "track_quality": round(float(max(0.0, min(1.0, quality))), 4),
            "failure_reason": "OK",
        })
    tracklets.sort(key=lambda x: x["start_frame"])
    return tracklets


__all__ = ["build_tracking_metrics", "build_tracklets"]
