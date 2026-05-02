"""Detection proxy metric builders — no ground truth required."""
from __future__ import annotations


def build_detection_metrics(
    total_frames: int,
    processed_frames: int,
    total_detections: int,
    confidence_sum: float,
    quality_sum: float,
    elapsed_sec: float,
    duration_sec: float,
    output_video_path: str,
    output_json_path: str,
) -> dict:
    avg_conf = confidence_sum / total_detections if total_detections > 0 else 0.0
    avg_quality = quality_sum / total_detections if total_detections > 0 else 0.0
    avg_per_frame = total_detections / processed_frames if processed_frames > 0 else 0.0
    fps_proc = processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0
    latency_ms = (elapsed_sec / processed_frames * 1000.0) if processed_frames > 0 else 0.0
    return {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "total_person_detections": total_detections,
        "avg_confidence": round(avg_conf, 4),
        "avg_detection_quality": round(avg_quality, 4),
        "avg_persons_per_frame": round(avg_per_frame, 3),
        "fps_processing": round(fps_proc, 2),
        "avg_latency_ms_per_frame": round(latency_ms, 2),
        "elapsed_sec": round(elapsed_sec, 3),
        "video_duration_sec": round(duration_sec, 3),
        "output_video_path": output_video_path,
        "output_json_path": output_json_path,
    }


__all__ = ["build_detection_metrics"]
