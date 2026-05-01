"""
Module 5 — Cross-Camera ReID API
=================================
Entry point: process_folder(multicam_dir, output_dir, pose_dir, adl_dir)

Reads all videos sorted by timestamp (per .algorithm.md §3.2),
then assigns permanent Global IDs (GID-001, GID-002, ...) that persist
across all cameras.  This is the ONLY module that renders GID labels on video.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from src.common.console import print_header, print_metric_table, print_saved
from src.common.metrics import Timer, load_json, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import create_video_writer, get_video_info, open_video
from src.common.visualization import draw_global_id, draw_skeleton
from src.global_reid.reid_core import GlobalPersonTable, list_sorted_clips, parse_clip


# ---------------------------------------------------------------------------
# Load ADL result for a given video stem from module 4 output
# ---------------------------------------------------------------------------

def _load_adl_events(adl_dir: Path | None, video_stem: str) -> dict[tuple[int, int], str]:
    """Return mapping (frame_id, track_id) -> adl_label from adl_events.json."""
    if adl_dir is None:
        return {}
    adl_json = adl_dir / video_stem / "adl_events.json"
    if not adl_json.exists():
        return {}
    events = load_json(adl_json, [])
    return {(int(e["frame_id"]), int(e.get("track_id", -1))): e.get("adl_label", "unknown") for e in events}


def _load_pose_records(pose_dir: Path | None, video_stem: str) -> list[dict]:
    if pose_dir is None:
        return []
    kp_json = pose_dir / video_stem / "keypoints.json"
    if not kp_json.exists():
        return []
    return load_json(kp_json, [])


# ---------------------------------------------------------------------------
# Process a single clip
# ---------------------------------------------------------------------------

def process_clip(
    clip_path: Path,
    cam_index: int,
    clip_dt: datetime,
    output_dir: Path,
    global_table: GlobalPersonTable,
    pose_dir: Path | None = None,
    adl_dir: Path | None = None,
) -> dict:
    """Process one video clip and update global_table in-place. Returns per-clip metrics."""
    video_stem = clip_path.stem
    clip_out = ensure_dir(output_dir / video_stem)
    overlay_path = clip_out / "reid_overlay.mp4"
    reid_json_path = clip_out / "reid_tracks.json"
    metric_path = clip_out / "reid_metrics.json"

    # Load upstream results
    pose_records = _load_pose_records(pose_dir, video_stem)
    adl_events = _load_adl_events(adl_dir, video_stem)

    info = get_video_info(clip_path)
    capture = open_video(clip_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)

    reid_records: list[dict] = []
    active_gids: set[int] = set()
    frame_id = 0
    timer = Timer()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # Get persons from pose keypoints (if available), else empty
            persons = []
            if frame_id < len(pose_records):
                persons = pose_records[frame_id].get("persons", [])

            frame_reid_results: list[dict] = []
            for person in persons:
                if not bool(person.get("is_confirmed", True)):
                    continue
                bbox = person.get("bbox", [0, 0, 0, 0])
                track_id = int(person.get("track_id") if person.get("track_id") is not None else -1)
                adl_label = adl_events.get((frame_id, track_id), "unknown")

                # Assign or retrieve Global ID
                gp, match_status = global_table.match_or_create(
                    bbox=bbox,
                    frame=frame,
                    cam_index=cam_index,
                    current_time=clip_dt,
                    adl_label=adl_label,
                )
                active_gids.add(gp.gid)

                # Draw Global ID on frame (ONLY this module does this)
                draw_global_id(frame, bbox, gp.gid, adl_label, match_status)

                # Draw skeleton
                draw_skeleton(frame, person.get("keypoints", []))

                frame_reid_results.append({
                    "track_id": track_id,
                    "global_id": gp.gid,
                    "match_status": match_status,
                    "adl_label": adl_label,
                    "bbox": bbox,
                    "failure_reason": "OK",
                })

            reid_records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                "cam_index": cam_index,
                "camera_id": f"cam{cam_index}",
                "persons": frame_reid_results,
                "failure_reason": "OK",
            })
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    # Mark persons missing from this clip
    global_table.mark_dormant_missing(active_gids, clip_dt)

    elapsed = timer.elapsed()
    metrics = {
        "clip": video_stem,
        "cam_index": cam_index,
        "clip_time": clip_dt.isoformat(),
        "total_frames": frame_id,
        "unique_global_ids": len(active_gids),
        "elapsed_sec": round(elapsed, 3),
        "fps_processing": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
        "overlay_path": str(overlay_path),
        "reid_json_path": str(reid_json_path),
        "metric_type": "proxy",
        "model_info": {"method": "tfcs-par-proxy"},
        "input_video": str(clip_path),
        "camera_id": f"cam{cam_index}",
        "start_time": clip_dt.isoformat(),
        "output_paths": {"overlay": str(overlay_path), "json": str(reid_json_path), "metrics": str(metric_path)},
        "failure_reason": "OK",
    }
    save_json(reid_json_path, reid_records)
    save_json(metric_path, metrics)
    print(
        f"  Frames: {frame_id} | GIDs active: {len(active_gids)} | "
        f"FPS: {metrics['fps_processing']:.2f} | cam{cam_index} @ {clip_dt.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print_saved(overlay_path, reid_json_path, metric_path)
    return metrics


# ---------------------------------------------------------------------------
# Main folder-level entry point
# ---------------------------------------------------------------------------

def process_folder(
    multicam_dir: str | Path,
    output_dir: str | Path,
    pose_dir: str | Path | None = None,
    adl_dir: str | Path | None = None,
) -> list[dict]:
    """Run cross-camera ReID over all clips sorted by timestamp."""
    multicam_dir = resolve_path(multicam_dir)
    output_dir = ensure_dir(resolve_path(output_dir))
    pose_dir_p = resolve_path(pose_dir) if pose_dir else None
    adl_dir_p = resolve_path(adl_dir) if adl_dir else None

    clips = list_sorted_clips(multicam_dir)

    print_header("CPose Cross-Camera ReID Module (TFCS-PAR)")
    print_metric_table({
        "Multicam folder": multicam_dir,
        "Output folder": output_dir,
        "Pose dir": pose_dir_p or "(not provided)",
        "ADL dir": adl_dir_p or "(not provided)",
        "Clips found (sorted)": len(clips),
    })

    if not clips:
        print("No parseable video clips found.")
        return []

    print("\nProcessing order (timestamp → cam):")
    for i, c in enumerate(clips, 1):
        print(f"  {i:>2}. cam{c.cam_index}  {c.clip_dt.strftime('%Y-%m-%d %H:%M:%S')}  {c.path.name}")

    global_table = GlobalPersonTable()
    results: list[dict] = []

    for idx, clip in enumerate(clips, 1):
        print(f"\n[{idx}/{len(clips)}] Processing {clip.path.name}")
        try:
            metrics = process_clip(
                clip_path=clip.path,
                cam_index=clip.cam_index,
                clip_dt=clip.clip_dt,
                output_dir=output_dir,
                global_table=global_table,
                pose_dir=pose_dir_p,
                adl_dir=adl_dir_p,
            )
            results.append(metrics)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # Save final Global Person Table
    table_path = output_dir / "global_person_table.json"
    save_json(table_path, global_table.to_dict())
    print(f"\nGlobal Person Table saved: {table_path}")

    # Summary
    all_gids: set[int] = set()
    for r in results:
        all_gids.update(range(1, r.get("unique_global_ids", 0) + 1))
    total_gids = global_table._next_gid - 1

    print_header("ReID SUMMARY")
    print_metric_table({
        "Total clips processed": len(results),
        "Total Global IDs assigned": total_gids,
        "Global Person Table": str(table_path),
    })
    return results
