from __future__ import annotations

from pathlib import Path

import cv2

from src.common.console import print_header, print_metric_table, print_saved, print_video_progress
from src.common.metrics import Timer, load_json, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video
from src.common.visualization import draw_track
from src.human_tracking.config import resolve_tracking_model
from src.human_tracking.metrics import build_tracking_metrics
from src.human_tracking.tracker import SimpleIoUTracker, YoloByteTracker


def _detection_json_for(video_path: Path, detection_dir: str | Path | None) -> Path | None:
    if detection_dir is None:
        return None
    candidate = resolve_path(detection_dir) / video_path.stem / "detections.json"
    return candidate if candidate.exists() else None


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
) -> dict:
    video_path = resolve_path(video_path)
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_output_dir / "tracking_overlay.mp4"
    json_path = video_output_dir / "tracks.json"
    metric_path = video_output_dir / "tracking_metrics.json"
    info = get_video_info(video_path)
    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)

    detection_json = _detection_json_for(video_path, detection_dir)
    detection_records = load_json(detection_json, []) if detection_json else []
    simple_tracker = SimpleIoUTracker(min_hits=min_hits, max_missing=max_age, window_size=window_size) if detection_records else None
    yolo_tracker = None if detection_records else YoloByteTracker(resolve_tracking_model(model), conf, tracker, min_hits, max_age, window_size)

    records: list[dict] = []
    frame_id = 0
    timer = Timer()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if simple_tracker is not None:
                detections = detection_records[frame_id]["detections"] if frame_id < len(detection_records) else []
                tracks = simple_tracker.update(detections)
            else:
                tracks = yolo_tracker.track(frame) if yolo_tracker else []
            for track_item in tracks:
                draw_track(frame, track_item["bbox"], track_item["track_id"], track_item["confidence"])
            records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                "camera_id": video_path.stem.split("_")[0],
                "tracks": tracks,
                "failure_reason": "OK",
            })
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    metrics = build_tracking_metrics(records, timer.elapsed(), str(overlay_path), str(json_path))
    metrics.update({
        "metric_type": "proxy",
        "model_info": {"model": str(resolve_tracking_model(model)), "tracker": tracker, "confidence": conf, "min_hits": min_hits, "max_age": max_age},
        "input_video": str(video_path),
        "camera_id": video_path.stem.split("_")[0],
        "start_time": None,
        "output_paths": {"overlay": str(overlay_path), "json": str(json_path), "metrics": str(metric_path)},
        "failure_reason": "OK",
    })
    save_json(json_path, records)
    save_json(metric_path, metrics)
    print(f"Frames: {len(records)} | Tracks: {metrics['total_tracks']} | FPS: {metrics['fps_processing']:.2f} | Latency: {metrics['avg_latency_ms_per_frame']:.2f} ms/frame")
    print_saved(overlay_path, json_path, metric_path)
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
) -> list[dict]:
    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)
    print_header("CPose Person Tracking Module")
    print_metric_table({
        "Input folder": resolve_path(input_dir),
        "Output folder": output_dir,
        "Model": resolve_tracking_model(model),
        "Tracker": tracker,
        "Videos found": len(videos),
    })
    if not videos:
        print("No videos found.")
        return []
    results = []
    for index, video in enumerate(videos, 1):
        print_video_progress(index, len(videos), video)
        try:
            results.append(process_video(video, output_dir, model, tracker, conf, detection_dir, min_hits, max_age, window_size))
        except Exception as exc:
            print(f"ERROR processing {video.name}: {exc}")
    print_header("SUMMARY")
    print_metric_table({
        "Total videos": len(results),
        "Total frames": sum(item.get("total_frames", 0) for item in results),
        "Total tracks": sum(item.get("total_tracks", 0) for item in results),
        "Average FPS": f"{sum(item.get('fps_processing', 0.0) for item in results) / len(results):.2f}" if results else "0.00",
    })
    return results
