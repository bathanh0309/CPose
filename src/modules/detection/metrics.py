"""Detection proxy metric builders."""
from __future__ import annotations


def build_detection_metrics(
    total_frames: int,
    processed_frames: int,
    total_detections: int,
    confidence_sum: float,
    elapsed_sec: float,
    video_duration_sec: float,
    output_video_path: str,
    output_json_path: str,
) -> dict:
    return {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "total_person_detections": total_detections,
        "avg_persons_per_frame": total_detections / processed_frames if processed_frames else 0.0,
        "avg_confidence": confidence_sum / total_detections if total_detections else 0.0,
        "fps_processing": processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
        "avg_latency_ms_per_frame": (elapsed_sec / processed_frames * 1000.0) if processed_frames else 0.0,
        "elapsed_sec": elapsed_sec,
        "video_duration_sec": video_duration_sec,
        "output_video_path": output_video_path,
        "output_json_path": output_json_path,
    }


__all__ = ["build_detection_metrics"]
