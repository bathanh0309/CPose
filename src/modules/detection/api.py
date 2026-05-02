"""Public API for CPose Module 1: detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.common.json_io import save_json
from src.common.logging_utils import print_header, print_metric_table, print_saved, print_video_progress
from src.common.paths import ensure_dir, resolve_path
from src.common.timer import Timer
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video, show_frame_preview
from src.common.visualization import draw_bbox
from src.modules.detection.detector import PersonDetector, resolve_detection_model
from src.modules.detection.metrics import build_detection_metrics


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
                "failure_reason": "OK" if detections else "NO_PERSON_DETECTED",
            })
            writer.write(frame)
            if preview:
                if show_frame_preview(f"CPose Detection | {video_path.stem}", frame, info.fps):
                    preview = False
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
    elapsed = timer.elapsed()
    metrics = build_detection_metrics(info.frame_count, frame_id, total_detections, confidence_sum, elapsed, info.duration_sec, str(overlay_path), str(json_path))
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
    print_metric_table({"Input folder": resolve_path(input_dir), "Output folder": output_dir, "Model": resolve_detection_model(model), "Videos found": len(videos)})
    if not videos:
        print("No videos found.")
        return []
    results = []
    for index, video in enumerate(videos, 1):
        print_video_progress(index, len(videos), video)
        try:
            results.append(process_video(video, output_dir, model, conf, preview))
        except Exception as exc:
            print(f"[ERROR] {video.name} | STEP_FAILED | reason={exc}")
    print_header("SUMMARY")
    print_metric_table({
        "Total videos": len(results),
        "Total frames": sum(item.get("processed_frames", 0) for item in results),
        "Total detections": sum(item.get("total_person_detections", 0) for item in results),
        "Average FPS": f"{sum(item.get('fps_processing', 0.0) for item in results) / len(results):.2f}" if results else "0.00",
    })
    return results


def run_detection(input_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> list[dict]:
    return process_folder(input_dir, output_dir, **kwargs)


__all__ = ["process_folder", "process_video", "run_detection"]
