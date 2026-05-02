"""Public API for CPose Module 2: local tracking — enhanced with trajectories, tracklets, and comparison."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.json_io import load_json, save_json
from src.common.logging_utils import print_header, print_metric_table, print_saved, print_video_progress
from src.common.paths import ensure_dir, resolve_path
from src.common.timer import Timer
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video, show_frame_preview
from src.common.visualization import draw_track
from src.modules.tracking.metrics import build_tracking_metrics, build_tracklets
from src.modules.tracking.tracker import SimpleIoUTracker, YoloByteTracker, resolve_tracking_model


# Per-track color palette (BGR) — deterministic from track_id
def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 200, 255), (255, 120, 0), (0, 255, 80), (200, 0, 255),
        (255, 255, 0), (0, 180, 255), (255, 60, 180), (80, 255, 200),
        (255, 200, 60), (120, 120, 255),
    ]
    return palette[track_id % len(palette)]


def _detection_json_for(video_path: Path, detection_dir: str | Path | None) -> Path | None:
    if detection_dir is None:
        return None
    candidate = resolve_path(detection_dir) / video_path.stem / "detections.json"
    return candidate if candidate.exists() else None


def _draw_trajectory_frame(
    frame: np.ndarray,
    track_history: dict[int, list[tuple[int, int]]],
    active_tracks: list[dict],
    camera_id: str,
    frame_id: int,
    max_points: int = 50,
) -> None:
    active_ids = {int(t["track_id"]) for t in active_tracks}
    # Draw trajectory lines for all tracks that have history
    for t_id, pts in track_history.items():
        if len(pts) < 2:
            continue
        color = _track_color(t_id)
        draw_pts = pts[-max_points:]
        for i in range(1, len(draw_pts)):
            alpha = i / len(draw_pts)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame, draw_pts[i - 1], draw_pts[i], c, 2)
        # Arrow for direction if enough points
        if len(draw_pts) >= 5:
            p1 = draw_pts[-5]
            p2 = draw_pts[-1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if dx*dx + dy*dy > 4:
                cv2.arrowedLine(frame, p1, p2, color, 2, tipLength=0.4)
    # HUD overlay
    n_active = len(active_ids)
    cv2.putText(frame, f"CPose Tracking | {camera_id} | f{frame_id} | active={n_active}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _make_trajectory_snapshot(
    background_frame: np.ndarray,
    track_history: dict[int, list[tuple[int, int]]],
    output_path: Path,
) -> None:
    canvas = background_frame.copy()
    overlay = canvas.copy()
    # Semi-transparent dark background for readability
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    for t_id, pts in track_history.items():
        if len(pts) < 2:
            continue
        color = _track_color(t_id)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 2)
        # Mark start and end
        cv2.circle(canvas, pts[0], 6, (0, 255, 0), -1)
        cv2.circle(canvas, pts[-1], 6, (0, 0, 255), -1)
        mid_idx = len(pts) // 2
        cv2.putText(canvas, f"T{t_id}", pts[mid_idx], cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _make_comparison_video(
    raw_path: Path,
    processed_path: Path,
    output_path: Path,
    left_label: str = "Raw Input",
    right_label: str = "CPose Tracking",
) -> None:
    cap_raw = cv2.VideoCapture(str(raw_path))
    cap_proc = cv2.VideoCapture(str(processed_path))
    fps = cap_raw.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ok1, f1 = cap_raw.read()
            ok2, f2 = cap_proc.read()
            if not ok1 or not ok2:
                break
            if f1.shape[:2] != (h, w):
                f1 = cv2.resize(f1, (w, h))
            if f2.shape[:2] != (h, w):
                f2 = cv2.resize(f2, (w, h))
            cv2.putText(f1, left_label, (12, 32), font, 0.9, (255, 255, 255), 2)
            cv2.putText(f1, left_label, (12, 32), font, 0.9, (0, 0, 0), 1)
            cv2.putText(f2, right_label, (12, 32), font, 0.9, (255, 255, 255), 2)
            cv2.putText(f2, right_label, (12, 32), font, 0.9, (0, 0, 0), 1)
            combined = cv2.hconcat([f1, f2])
            writer.write(combined)
    finally:
        cap_raw.release()
        cap_proc.release()
        writer.release()


def _enrich_track_with_kinematics(track: dict, track_history: list[tuple[int, int]], fps: float) -> dict:
    """Add velocity, speed, and direction to a track dict."""
    enriched = dict(track)
    bbox = track.get("bbox", [0, 0, 0, 0])
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bcx, bcy = cx, bbox[3]
    enriched["center"] = [round(cx, 1), round(cy, 1)]
    enriched["bottom_center"] = [round(bcx, 1), round(bcy, 1)]

    vx, vy = 0.0, 0.0
    speed = 0.0
    direction = 0.0
    if len(track_history) >= 2:
        prev = track_history[-2]
        curr = track_history[-1]
        vx = float(curr[0] - prev[0]) * fps
        vy = float(curr[1] - prev[1]) * fps
        speed = math.sqrt(vx * vx + vy * vy)
        direction = math.degrees(math.atan2(vy, vx)) % 360.0
    enriched["velocity"] = [round(vx, 2), round(vy, 2)]
    enriched["speed_px_per_sec"] = round(speed, 2)
    enriched["direction_deg"] = round(direction, 1)
    enriched["trajectory_length"] = len(track_history)
    enriched["local_track_id"] = f"T{int(track['track_id']):03d}"
    return enriched


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    model: str | Path | None = None,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.5,
    detection_dir: str | Path | None = None,
    min_hits: int = 3,
    max_age: int = 30,
    window_size: int = 30,
    preview: bool = True,
    make_comparison: bool = False,
    comparison_dir: str | Path | None = None,
) -> dict:
    video_path = resolve_path(video_path)
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_output_dir / "tracking_overlay.mp4"
    traj_overlay_path = video_output_dir / "trajectory_overlay.mp4"
    traj_snapshot_path = video_output_dir / "trajectory_snapshot.png"
    json_path = video_output_dir / "tracks.json"
    tracklets_path = video_output_dir / "tracklets.json"
    metric_path = video_output_dir / "tracking_metrics.json"

    info = get_video_info(video_path)
    fps = info.fps or 25.0
    camera_id = video_path.stem.split("_")[0]

    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, fps, info.width, info.height)
    traj_writer = create_video_writer(traj_overlay_path, fps, info.width, info.height)

    detection_json = _detection_json_for(video_path, detection_dir)
    detection_records = load_json(detection_json, []) if detection_json else []
    simple_tracker = SimpleIoUTracker(min_hits=min_hits, max_missing=max_age, window_size=window_size) if detection_records else None
    yolo_tracker = None if detection_records else YoloByteTracker(resolve_tracking_model(model), conf, tracker, min_hits, max_age, window_size)

    records: list[dict] = []
    frame_id = 0
    timer = Timer()
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    background_frame: np.ndarray | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if background_frame is None:
                background_frame = frame.copy()

            if simple_tracker is not None:
                detections = detection_records[frame_id]["detections"] if frame_id < len(detection_records) else []
                raw_tracks = simple_tracker.update(detections)
            else:
                raw_tracks = yolo_tracker.track(frame) if yolo_tracker else []

            tracks = []
            traj_frame = frame.copy()

            for track_item in raw_tracks:
                t_id = int(track_item["track_id"])
                bbox = track_item["bbox"]
                bc = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                track_history[t_id].append(bc)
                if len(track_history[t_id]) > 120:
                    track_history[t_id].pop(0)

                enriched = _enrich_track_with_kinematics(track_item, track_history[t_id], fps)
                tracks.append(enriched)

                # Draw on tracking overlay
                draw_track(frame, bbox, t_id, track_item["confidence"], track_history=track_history[t_id][-30:])

            # Draw trajectory overlay
            _draw_trajectory_frame(traj_frame, track_history, tracks, camera_id, frame_id)

            records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / fps if fps > 0 else 0.0,
                "camera_id": camera_id,
                "tracks": tracks,
                "failure_reason": "OK" if tracks else "NO_PERSON_DETECTED",
            })
            writer.write(frame)
            traj_writer.write(traj_frame)

            if preview:
                if show_frame_preview(f"CPose Tracking | {video_path.stem}", frame, fps):
                    preview = False
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        traj_writer.release()
        cv2.destroyAllWindows()

    # Trajectory snapshot
    if background_frame is not None:
        _make_trajectory_snapshot(background_frame, track_history, traj_snapshot_path)

    # Build tracklets
    tracklets = build_tracklets(records, fps, camera_id, track_history)
    save_json(tracklets_path, tracklets)

    metrics = build_tracking_metrics(
        records, timer.elapsed(),
        str(overlay_path), str(json_path),
        trajectory_video_path=str(traj_overlay_path),
        trajectory_snapshot_path=str(traj_snapshot_path),
        tracklets_json_path=str(tracklets_path),
    )
    metrics.update({
        "metric_type": "proxy",
        "model_info": {
            "model": str(resolve_tracking_model(model)),
            "tracker": tracker,
            "confidence": conf,
            "min_hits": min_hits,
            "max_age": max_age,
        },
        "input_video": str(video_path),
        "camera_id": camera_id,
        "start_time": None,
        "output_paths": {
            "overlay": str(overlay_path),
            "trajectory_overlay": str(traj_overlay_path),
            "trajectory_snapshot": str(traj_snapshot_path),
            "json": str(json_path),
            "tracklets": str(tracklets_path),
            "metrics": str(metric_path),
        },
        "failure_reason": "OK",
    })
    save_json(json_path, records)
    save_json(metric_path, metrics)

    confirmed = metrics.get("confirmed_tracks", 0)
    total = metrics.get("total_tracks", 0)
    frag = metrics.get("proxy_track_fragmentation", 0)
    print(
        f"Frames={len(records)} | Tracks={total} | Confirmed={confirmed} "
        f"| FragProxy={frag} | FPS={metrics['fps_processing']:.2f}"
    )
    print_saved(overlay_path, json_path, metric_path)

    if make_comparison and comparison_dir:
        comp_dir = ensure_dir(resolve_path(comparison_dir))
        for label, proc_path in [
            ("tracking", overlay_path),
            ("trajectory", traj_overlay_path),
        ]:
            comp_path = comp_dir / f"{video_path.stem}_raw_vs_{label}.mp4"
            try:
                _make_comparison_video(video_path, proc_path, comp_path, "Raw Input", f"CPose {label.title()}")
                print(f"Comparison: {comp_path}")
            except Exception as exc:
                print(f"[WARN] Comparison video failed ({label}): {exc}")

    return metrics


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    model: str | Path | None = None,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.5,
    detection_dir: str | Path | None = None,
    min_hits: int = 3,
    max_age: int = 30,
    window_size: int = 30,
    preview: bool = True,
    make_comparison: bool = False,
    compare_count: int = 2,
    comparison_dir: str | Path | None = None,
) -> list[dict]:
    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)
    if comparison_dir is None:
        comparison_dir = resolve_path(output_dir).parent / "06_comparison"
    print_header("CPose Person Tracking Module")
    print_metric_table({
        "Input folder": resolve_path(input_dir),
        "Output folder": output_dir,
        "Model": resolve_tracking_model(model),
        "Tracker": tracker,
        "Videos found": len(videos),
        "Comparison": f"enabled, first {compare_count}" if make_comparison else "disabled",
    })
    results = []
    for index, video in enumerate(videos, 1):
        print_video_progress(index, len(videos), video)
        do_compare = make_comparison and index <= compare_count
        try:
            results.append(process_video(
                video, output_dir, model, tracker, conf,
                detection_dir, min_hits, max_age, window_size,
                preview, do_compare, comparison_dir,
            ))
        except Exception as exc:
            print(f"[ERROR] {video.name} | STEP_FAILED | reason={exc}")

    print_header("SUMMARY")
    total_t = sum(item.get("total_tracks", 0) for item in results)
    total_c = sum(item.get("confirmed_tracks", 0) for item in results)
    ratio = total_c / total_t if total_t > 0 else 0.0
    fps_vals = [item.get("fps_processing", 0.0) for item in results if item.get("fps_processing")]
    avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0
    print_metric_table({
        "Total videos": len(results),
        "Total frames": sum(item.get("total_frames", 0) for item in results),
        "Total tracks": total_t,
        "Confirmed tracks": total_c,
        "Confirmed track ratio": f"{ratio:.3f}",
        "Average FPS": f"{avg_fps:.2f}",
        "Metric type": "proxy",
    })
    return results


def run_tracking(input_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> list[dict]:
    return process_folder(input_dir, output_dir, **kwargs)


__all__ = ["process_folder", "process_video", "run_tracking"]
