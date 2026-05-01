from __future__ import annotations

from pathlib import Path

import cv2

from src.common.console import print_header, print_metric_table, print_saved, print_video_progress
from src.common.metrics import Timer, load_json, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video
from src.common.visualization import draw_bbox, draw_skeleton
from src.human_tracking.tracker import iou
from src.pose_estimation.config import resolve_pose_model
from src.pose_estimation.metrics import build_pose_metrics
from src.pose_estimation.pose_model import PoseModel


def _track_json_for(video_path: Path, track_dir: str | Path | None) -> Path | None:
    if track_dir is None:
        return None
    candidate = resolve_path(track_dir) / video_path.stem / "tracks.json"
    return candidate if candidate.exists() else None


def _assign_track_ids(
    persons: list[dict],
    tracks: list[dict],
    threshold: float = 0.3,
    run_on_confirmed_tracks_only: bool = True,
) -> None:
    used: set[int] = set()
    eligible_tracks = [track for track in tracks if bool(track.get("is_confirmed", True))] if run_on_confirmed_tracks_only else tracks
    for person in persons:
        best_track = None
        best_score = 0.0
        for track in eligible_tracks:
            track_id = int(track["track_id"])
            if track_id in used:
                continue
            score = iou(person["bbox"], track["bbox"])
            if score > best_score:
                best_score = score
                best_track = track
        if best_track and best_score >= threshold:
            person["track_id"] = int(best_track["track_id"])
            person["is_confirmed"] = bool(best_track.get("is_confirmed", True))
            person["pose_track_iou"] = best_score
            if person.get("failure_reason") != "LOW_KEYPOINT_VISIBILITY":
                person["failure_reason"] = best_track.get("failure_reason", "OK")
            used.add(int(best_track["track_id"]))
        else:
            person["track_id"] = None
            person["is_confirmed"] = False
            person["pose_track_iou"] = None
            person["failure_reason"] = "NO_MATCHED_TRACK"


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    model: str | Path | None = None,
    conf: float = 0.5,
    track_dir: str | Path | None = None,
    track_iou_threshold: float = 0.3,
    min_visible_keypoints: int = 8,
    keypoint_conf: float = 0.30,
    run_on_confirmed_tracks_only: bool = True,
) -> dict:
    video_path = resolve_path(video_path)
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_output_dir / "pose_overlay.mp4"
    json_path = video_output_dir / "keypoints.json"
    metric_path = video_output_dir / "pose_metrics.json"
    pose_model = PoseModel(resolve_pose_model(model), conf, keypoint_conf=keypoint_conf, min_visible_keypoints=min_visible_keypoints)
    info = get_video_info(video_path)
    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)
    track_json = _track_json_for(video_path, track_dir)
    track_records = load_json(track_json, []) if track_json else []
    records: list[dict] = []
    frame_id = 0
    timer = Timer()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            persons = pose_model.estimate(frame)
            tracks = track_records[frame_id]["tracks"] if frame_id < len(track_records) else []
            _assign_track_ids(persons, tracks, track_iou_threshold, run_on_confirmed_tracks_only)
            for person in persons:
                # Module 3 (Pose): do NOT render Global IDs — that is ReID's job (module 5).
                # track_id here is a local-clip tracker ID stored in JSON for downstream use only.
                draw_bbox(frame, person["bbox"], "person", (0, 200, 255))
                draw_skeleton(frame, person["keypoints"])
            records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                "camera_id": video_path.stem.split("_")[0],
                "persons": persons,
                "failure_reason": "OK",
            })
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
    metrics = build_pose_metrics(records, timer.elapsed(), str(overlay_path), str(json_path))
    metrics.update({
        "metric_type": "proxy",
        "model_info": {"model": str(resolve_pose_model(model)), "confidence": conf, "keypoint_conf": keypoint_conf, "track_iou_threshold": track_iou_threshold},
        "input_video": str(video_path),
        "camera_id": video_path.stem.split("_")[0],
        "start_time": None,
        "output_paths": {"overlay": str(overlay_path), "json": str(json_path), "metrics": str(metric_path)},
        "failure_reason": "OK",
    })
    save_json(json_path, records)
    save_json(metric_path, metrics)
    print(f"Frames: {len(records)} | Poses: {metrics['total_pose_instances']} | FPS: {metrics['fps_processing']:.2f} | Latency: {metrics['avg_latency_ms_per_frame']:.2f} ms/frame")
    print_saved(overlay_path, json_path, metric_path)
    return metrics


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    model: str | Path | None = None,
    conf: float = 0.5,
    track_dir: str | Path | None = None,
    track_iou_threshold: float = 0.3,
    min_visible_keypoints: int = 8,
    keypoint_conf: float = 0.30,
    run_on_confirmed_tracks_only: bool = True,
) -> list[dict]:
    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)
    print_header("CPose Pose Estimation Module")
    print_metric_table({
        "Input folder": resolve_path(input_dir),
        "Output folder": output_dir,
        "Model": resolve_pose_model(model),
        "Videos found": len(videos),
    })
    if not videos:
        print("No videos found.")
        return []
    results = []
    for index, video in enumerate(videos, 1):
        print_video_progress(index, len(videos), video)
        try:
            results.append(process_video(
                video,
                output_dir,
                model,
                conf,
                track_dir,
                track_iou_threshold,
                min_visible_keypoints,
                keypoint_conf,
                run_on_confirmed_tracks_only,
            ))
        except Exception as exc:
            print(f"ERROR processing {video.name}: {exc}")
    print_header("SUMMARY")
    print_metric_table({
        "Total videos": len(results),
        "Total frames": sum(item.get("total_frames", 0) for item in results),
        "Total pose instances": sum(item.get("total_pose_instances", 0) for item in results),
        "Average FPS": f"{sum(item.get('fps_processing', 0.0) for item in results) / len(results):.2f}" if results else "0.00",
    })
    return results
