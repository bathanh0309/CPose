"""Public API for CPose Module 1: detection — enhanced with crops and evidence."""
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
from src.modules.detection.detector import PersonDetector, reset_detection_counter, resolve_detection_model
from src.modules.detection.metrics import build_detection_metrics


def _make_comparison_video(
    raw_path: Path,
    processed_path: Path,
    output_path: Path,
    left_label: str = "Raw Input",
    right_label: str = "CPose Detection",
) -> None:
    """Create side-by-side comparison video."""
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


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    model: str | Path | None = None,
    conf: float = 0.5,
    preview: bool = True,
    save_crops: bool = True,
    make_comparison: bool = False,
    comparison_dir: str | Path | None = None,
) -> dict:
    video_path = resolve_path(video_path)
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_output_dir / "detection_overlay.mp4"
    json_path = video_output_dir / "detections.json"
    metric_path = video_output_dir / "detection_metrics.json"
    crops_dir = video_output_dir / "crops" if save_crops else None
    model_path = resolve_detection_model(model)
    detector = PersonDetector(model_path, conf)
    info = get_video_info(video_path)
    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)
    frame_id = 0
    total_detections = 0
    confidence_sum = 0.0
    quality_sum = 0.0
    records: list[dict] = []
    timer = Timer()
    camera_id = video_path.stem.split("_")[0]
    reset_detection_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame, crops_dir=crops_dir, frame_id=frame_id, camera_id=camera_id)
            for det in detections:
                draw_bbox(frame, det["bbox"], f"person {det['confidence']:.2f}")
                total_detections += 1
                confidence_sum += det["confidence"]
                quality_sum += det.get("detection_quality", det["confidence"])
            records.append({
                "frame_id": frame_id,
                "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                "camera_id": camera_id,
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
    metrics = build_detection_metrics(
        info.frame_count, frame_id, total_detections,
        confidence_sum, quality_sum, elapsed, info.duration_sec,
        str(overlay_path), str(json_path),
    )
    metrics.update({
        "metric_type": "proxy",
        "model_info": {"model": str(model_path), "confidence": conf},
        "input_video": str(video_path),
        "camera_id": camera_id,
        "start_time": None,
        "output_paths": {
            "overlay": str(overlay_path),
            "json": str(json_path),
            "metrics": str(metric_path),
            "crops": str(crops_dir) if crops_dir else None,
        },
        "failure_reason": "OK",
    })
    save_json(json_path, records)
    save_json(metric_path, metrics)
    print(f"Frames: {frame_id} | Persons: {total_detections} | AvgQuality: {metrics['avg_detection_quality']:.3f} | FPS: {metrics['fps_processing']:.2f} | Latency: {metrics['avg_latency_ms_per_frame']:.2f} ms/frame")
    print_saved(overlay_path, json_path, metric_path)

    if make_comparison and comparison_dir:
        comp_dir = ensure_dir(resolve_path(comparison_dir))
        comp_path = comp_dir / f"{video_path.stem}_raw_vs_detection.mp4"
        try:
            _make_comparison_video(video_path, overlay_path, comp_path, "Raw Input", "CPose Detection")
            print(f"Comparison: {comp_path}")
        except Exception as exc:
            print(f"[WARN] Comparison video failed for {video_path.stem}: {exc}")

    return metrics


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    model: str | Path | None = None,
    conf: float = 0.5,
    preview: bool = True,
    save_crops: bool = True,
    make_comparison: bool = False,
    compare_count: int = 2,
    comparison_dir: str | Path | None = None,
) -> list[dict]:
    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)
    if comparison_dir is None:
        comparison_dir = resolve_path(output_dir).parent / "06_comparison"
    print_header("CPose Person Detection Module")
    print_metric_table({
        "Input folder": resolve_path(input_dir),
        "Output folder": output_dir,
        "Model": resolve_detection_model(model),
        "Videos found": len(videos),
        "Save crops": save_crops,
        "Comparison": f"enabled, first {compare_count}" if make_comparison else "disabled",
    })
    if not videos:
        print("No videos found.")
        return []
    results = []
    for index, video in enumerate(videos, 1):
        print_video_progress(index, len(videos), video)
        do_compare = make_comparison and index <= compare_count
        try:
            results.append(process_video(video, output_dir, model, conf, preview, save_crops, do_compare, comparison_dir))
        except Exception as exc:
            print(f"[ERROR] {video.name} | STEP_FAILED | reason={exc}")
    print_header("SUMMARY")
    print_metric_table({
        "Total videos": len(results),
        "Total frames": sum(item.get("processed_frames", 0) for item in results),
        "Total detections": sum(item.get("total_person_detections", 0) for item in results),
        "Avg detection quality": f"{sum(item.get('avg_detection_quality', 0.0) for item in results) / len(results):.3f}" if results else "N/A",
        "Average FPS": f"{sum(item.get('fps_processing', 0.0) for item in results) / len(results):.2f}" if results else "0.00",
    })
    return results


def run_detection(input_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> list[dict]:
    return process_folder(input_dir, output_dir, **kwargs)


__all__ = ["process_folder", "process_video", "run_detection"]
