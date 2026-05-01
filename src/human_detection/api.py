from __future__ import annotations

from pathlib import Path

import cv2

from src.common.console import print_header, print_metric_table, print_saved, print_video_progress
from src.common.metrics import Timer, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video, show_frame_preview
from src.common.visualization import draw_bbox
from src.human_detection.config import resolve_detection_model
from src.human_detection.detector import PersonDetector
from src.human_detection.metrics import build_detection_metrics


def process_video(video_path: str | Path, output_dir: str | Path, model: str | Path | None = None, conf: float = 0.5, preview: bool = True) -> dict:
    video_path = resolve_path(video_path)
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_output_dir / "detection_overlay.mp4"
    json_path = video_output_dir / "detections.json"
    metric_path = video_output_dir / "detection_metrics.json"

    model_path = resolve_detection_model(model)
    detector = PersonDetector(model_path, conf)
    info = get_video_info(video_path)
    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)

    frame_id = 0
    total_detections = 0
    confidence_sum = 0.0
    records: list[dict] = []
    timer = Timer()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            for detection in detections:
                draw_bbox(frame, detection["bbox"], f"person {detection['confidence']:.2f}")
                total_detections += 1
                confidence_sum += detection["confidence"]
            records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                "camera_id": video_path.stem.split("_")[0],
                "detections": detections,
                "failure_reason": "OK",
            })
            writer.write(frame)
            if preview:
                show_frame_preview(f"CPose Detection | {video_path.stem}", frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyWindow(f"CPose Detection | {video_path.stem}")

    elapsed = timer.elapsed()
    metrics = build_detection_metrics(
        info.frame_count,
        frame_id,
        total_detections,
        confidence_sum,
        elapsed,
        info.duration_sec,
        str(overlay_path),
        str(json_path),
    )
    metrics.update({
        "metric_type": "proxy",
        "model_info": {"model": str(model_path), "confidence": conf},
        "input_video": str(video_path),
        "camera_id": video_path.stem.split("_")[0],
        "start_time": None,
        "output_paths": {"overlay": str(overlay_path), "json": str(json_path), "metrics": str(metric_path)},
        "failure_reason": "OK",
    })
    save_json(json_path, records)
    save_json(metric_path, metrics)
    print(f"Frames: {frame_id} | Persons: {total_detections} | FPS: {metrics['fps_processing']:.2f} | Latency: {metrics['avg_latency_ms_per_frame']:.2f} ms/frame")
    print_saved(overlay_path, json_path, metric_path)
    return metrics


def process_folder(input_dir: str | Path, output_dir: str | Path, model: str | Path | None = None, conf: float = 0.5, preview: bool = True) -> list[dict]:
    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)
    print_header("CPose Person Detection Module")
    print_metric_table({
        "Input folder": resolve_path(input_dir),
        "Output folder": output_dir,
        "Model": resolve_detection_model(model),
        "Videos found": len(videos),
    })
    if not videos:
        print("No videos found.")
        return []
    results = []
    for index, video in enumerate(videos, 1):
        print_video_progress(index, len(videos), video)
        try:
            results.append(process_video(video, output_dir, model, conf, preview))
        except Exception as exc:
            print(f"ERROR processing {video.name}: {exc}")
    print_header("SUMMARY")
    total_frames = sum(item.get("processed_frames", 0) for item in results)
    total_detections = sum(item.get("total_person_detections", 0) for item in results)
    avg_fps = sum(item.get("fps_processing", 0.0) for item in results) / len(results) if results else 0.0
    avg_latency = sum(item.get("avg_latency_ms_per_frame", 0.0) for item in results) / len(results) if results else 0.0
    print_metric_table({
        "Total videos": len(results),
        "Total frames": total_frames,
        "Total detections": total_detections,
        "Average FPS": f"{avg_fps:.2f}",
        "Average latency/frame": f"{avg_latency:.2f} ms",
    })
    return results
