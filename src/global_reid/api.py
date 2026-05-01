from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from src.common.console import print_header, print_metric_table, print_saved
from src.common.manifest import ResolvedVideoItem, resolve_videos_from_manifest
from src.common.metrics import Timer, load_json, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.topology import CameraTopology, load_camera_topology
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video
from src.common.visualization import draw_global_id, draw_skeleton
from src.global_reid.reid_core import GlobalPersonTable


def _load_adl_events(adl_dir: Path | None, video_stem: str) -> dict[tuple[int, int], str]:
    if adl_dir is None:
        return {}
    adl_json = adl_dir / video_stem / "adl_events.json"
    if not adl_json.exists():
        return {}
    events = load_json(adl_json, [])
    return {(int(e["frame_id"]), int(e.get("track_id", -1))): e.get("adl_label", "unknown") for e in events}


def _load_face_events(face_dir: Path | None, video_stem: str) -> dict[tuple[int, int], dict]:
    if face_dir is None:
        return {}
    face_json = face_dir / video_stem / "face_events.json"
    if not face_json.exists():
        return {}
    events = load_json(face_json, [])
    return {(int(e["frame_id"]), int(e.get("track_id", -1))): e for e in events}


def _load_pose_records(pose_dir: Path | None, video_stem: str) -> list[dict]:
    if pose_dir is None:
        return []
    kp_json = pose_dir / video_stem / "keypoints.json"
    return load_json(kp_json, []) if kp_json.exists() else []


def _fallback_items(multicam_dir: Path) -> list[ResolvedVideoItem]:
    return resolve_videos_from_manifest(multicam_dir, None)


def process_clip(
    item: ResolvedVideoItem,
    output_dir: Path,
    global_table: GlobalPersonTable,
    topology: CameraTopology,
    config: dict[str, Any] | None = None,
    pose_dir: Path | None = None,
    adl_dir: Path | None = None,
    face_dir: Path | None = None,
) -> dict[str, Any]:
    config = config or {}
    clip_path = item.path
    video_stem = item.stem
    clip_out = ensure_dir(output_dir / video_stem)
    overlay_path = clip_out / "reid_overlay.mp4"
    reid_json_path = clip_out / "reid_tracks.json"
    metric_path = clip_out / "reid_metrics.json"
    pose_records = _load_pose_records(pose_dir, video_stem)
    adl_events = _load_adl_events(adl_dir, video_stem)
    face_events = _load_face_events(face_dir, video_stem)
    info = get_video_info(clip_path)
    capture = open_video(clip_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)
    reid_records: list[dict[str, Any]] = []
    active_gids: set[int] = set()
    frame_id = 0
    score_values: list[float] = []
    failure_distribution: Counter[str] = Counter()
    match_distribution: Counter[str] = Counter()
    timer = Timer()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            persons = pose_records[frame_id].get("persons", []) if frame_id < len(pose_records) else []
            frame_results: list[dict[str, Any]] = []
            current_time = item.start_time + timedelta(seconds=frame_id / info.fps) if item.start_time and info.fps > 0 else item.start_time
            for person in persons:
                if not bool(person.get("is_confirmed", True)):
                    continue
                bbox = person.get("bbox", [0, 0, 0, 0])
                track_id = int(person.get("track_id") if person.get("track_id") is not None else -1)
                adl_label = adl_events.get((frame_id, track_id))
                face_event = face_events.get((frame_id, track_id))
                gp, match_info = global_table.match_or_create(
                    bbox=bbox,
                    frame=frame,
                    camera_id=item.camera_id,
                    current_time=current_time,
                    track_id=track_id,
                    adl_label=adl_label,
                    keypoints=person.get("keypoints"),
                    face_event=face_event,
                    topology=topology,
                    config=config,
                )
                active_gids.add(gp.gid)
                draw_global_id(frame, bbox, gp.gid, adl_label or "unknown", gp.state)
                draw_skeleton(frame, person.get("keypoints", []))
                if match_info.get("score_total") is not None:
                    score_values.append(float(match_info["score_total"]))
                failure_distribution.update([match_info.get("failure_reason", "OK")])
                match_distribution.update([match_info.get("match_status", "unknown")])
                frame_results.append({
                    "local_track_id": track_id,
                    "global_id": gp.global_id,
                    "state": match_info.get("state", gp.state),
                    "match_status": match_info.get("match_status"),
                    "bbox": bbox,
                    "adl_label": adl_label,
                    "score_total": match_info.get("score_total"),
                    "score_face": match_info.get("score_face"),
                    "score_body": match_info.get("score_body"),
                    "score_pose": match_info.get("score_pose"),
                    "score_height": match_info.get("score_height"),
                    "score_time": match_info.get("score_time"),
                    "score_topology": match_info.get("score_topology"),
                    "topology_allowed": match_info.get("topology_allowed"),
                    "delta_time_sec": match_info.get("delta_time_sec"),
                    "entry_zone": match_info.get("entry_zone"),
                    "exit_zone": match_info.get("exit_zone"),
                    "failure_reason": match_info.get("failure_reason", "OK"),
                })
            reid_records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                "camera_id": item.camera_id,
                "persons": frame_results,
                "failure_reason": "OK",
            })
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    global_table.mark_dormant_missing(active_gids, item.start_time, float(config.get("max_candidate_age_sec", 300)))
    elapsed = timer.elapsed()
    metrics = {
        "metric_type": "proxy",
        "clip": video_stem,
        "total_frames": frame_id,
        "processed_frames": frame_id,
        "unique_global_ids": len(active_gids),
        "elapsed_sec": round(elapsed, 3),
        "fps_processing": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_latency_ms_per_frame": (elapsed / frame_id * 1000.0) if frame_id else 0.0,
        "avg_score_total": sum(score_values) / len(score_values) if score_values else None,
        "strong_match_count": match_distribution.get("strong_match", 0),
        "soft_match_count": match_distribution.get("soft_match", 0),
        "ambiguous_match_count": match_distribution.get("ambiguous", 0),
        "new_id_count": match_distribution.get("new_id", 0),
        "topology_conflict_count": failure_distribution.get("TOPOLOGY_CONFLICT", 0),
        "multi_candidate_conflict_count": failure_distribution.get("MULTI_CANDIDATE_CONFLICT", 0),
        "failure_reason_distribution": dict(failure_distribution),
        "model_info": {"method": config.get("method", "tfcs_par"), "weights": config.get("weights")},
        "input_video": str(clip_path),
        "camera_id": item.camera_id,
        "start_time": item.start_time.isoformat() if item.start_time else None,
        "output_paths": {"overlay": str(overlay_path), "json": str(reid_json_path), "metrics": str(metric_path)},
        "failure_reason": "OK",
    }
    save_json(reid_json_path, reid_records)
    save_json(metric_path, metrics)
    print_saved(overlay_path, reid_json_path, metric_path)
    return metrics


def process_folder(
    multicam_dir: str | Path,
    output_dir: str | Path,
    pose_dir: str | Path | None = None,
    adl_dir: str | Path | None = None,
    face_dir: str | Path | None = None,
    manifest: str | Path | list[ResolvedVideoItem] | None = None,
    topology: str | Path | CameraTopology | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    multicam_dir = resolve_path(multicam_dir)
    output_dir = ensure_dir(resolve_path(output_dir))
    pose_dir_p = resolve_path(pose_dir) if pose_dir else None
    adl_dir_p = resolve_path(adl_dir) if adl_dir else None
    face_dir_p = resolve_path(face_dir) if face_dir else None
    items = manifest if isinstance(manifest, list) else resolve_videos_from_manifest(multicam_dir, Path(manifest) if manifest else None)
    if not items:
        items = _fallback_items(multicam_dir)
    topology_obj = topology if isinstance(topology, CameraTopology) else load_camera_topology(topology)
    print_header("CPose Cross-Camera ReID Module")
    print_metric_table({
        "Multicam folder": multicam_dir,
        "Output folder": output_dir,
        "Pose dir": pose_dir_p or "(not provided)",
        "ADL dir": adl_dir_p or "(not provided)",
        "Face dir": face_dir_p or "(not provided)",
        "Clips found": len(items),
    })
    if not items:
        print("No parseable video clips found.")
        return []
    global_table = GlobalPersonTable()
    results: list[dict] = []
    for item in items:
        if not item.path.exists():
            print(f"[WARN] ReID input video missing: {item.path}")
            continue
        try:
            results.append(process_clip(item, output_dir, global_table, topology_obj, config, pose_dir_p, adl_dir_p, face_dir_p))
        except Exception as exc:
            print(f"ERROR processing ReID for {item.path.name}: {exc}")
            err_dir = ensure_dir(output_dir / item.stem)
            save_json(err_dir / "error.json", {"failure_reason": "STEP_FAILED", "error": str(exc)})
    table_path = output_dir / "global_person_table.json"
    save_json(table_path, global_table.to_dict())
    summary_path = output_dir / "reid_metrics.json"
    score_values = [row.get("avg_score_total") for row in results if row.get("avg_score_total") is not None]
    summary = {
        "metric_type": "proxy",
        "total_global_ids": len(global_table.to_dict()),
        **global_table.counts,
        "avg_score_total": sum(score_values) / len(score_values) if score_values else None,
        "failure_reason_distribution": dict(Counter(reason for row in results for reason, count in row.get("failure_reason_distribution", {}).items() for _ in range(count))),
        "failure_reason": "OK",
    }
    save_json(summary_path, summary)
    return results
