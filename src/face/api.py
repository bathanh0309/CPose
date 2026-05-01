from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.console import print_header, print_metric_table, print_saved
from src.common.metrics import Timer, load_json, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import get_video_info, list_video_files, open_video
from src.face.anti_spoofing import OptionalAntiSpoofing
from src.face.face_detector import OptionalFaceDetector
from src.face.face_recognizer import OptionalFaceRecognizer
from src.face.metrics import build_face_metrics


def _track_json_for(video_path: Path, track_dir: str | Path | None) -> Path | None:
    if track_dir is None:
        return None
    candidate = resolve_path(track_dir) / video_path.stem / "tracks.json"
    return candidate if candidate.exists() else None


def _crop(frame: Any, bbox: list[float]) -> Any:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    track_dir: str | Path | None = None,
    model_config: dict | None = None,
    run_every_n_frames: int = 10,
) -> dict[str, Any]:
    config = model_config or {}
    video_path = resolve_path(video_path)
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    events_path = video_output_dir / "face_events.json"
    metric_path = video_output_dir / "face_metrics.json"
    detector = OptionalFaceDetector(min_face_size=int(config.get("min_face_size", 40)))
    recognizer = OptionalFaceRecognizer(enabled=bool(config.get("recognizer")))
    anti_spoof = OptionalAntiSpoofing(bool(config.get("anti_spoofing_enabled", False)), config.get("anti_spoofing_model"))
    track_json = _track_json_for(video_path, track_dir)
    track_records = load_json(track_json, []) if track_json else []
    info = get_video_info(video_path)
    capture = open_video(video_path)
    events: list[dict[str, Any]] = []
    frame_id = 0
    timer = Timer()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id % max(1, run_every_n_frames) != 0:
                frame_id += 1
                continue
            tracks = track_records[frame_id].get("tracks", []) if frame_id < len(track_records) else []
            for track in tracks:
                if not bool(track.get("is_confirmed", True)):
                    continue
                track_crop = _crop(frame, track.get("bbox", []))
                local_face_bbox, face_quality, reason = detector.detect(track_crop)
                face_detected = local_face_bbox is not None
                embedding, embedding_dim, embed_reason = (None, None, "NO_FACE")
                spoof_status, spoof_reason = "unchecked", "ANTI_SPOOF_DISABLED"
                if face_detected and track_crop is not None:
                    embedding, embedding_dim, embed_reason = recognizer.embed(track_crop)
                    spoof_status, spoof_reason = anti_spoof.check(track_crop)
                final_reason = "OK" if face_detected else reason
                if face_detected and embedding is None and embed_reason == "MODEL_MISSING":
                    final_reason = "OK"
                if spoof_reason == "MODEL_MISSING" and config.get("anti_spoofing_enabled", False):
                    final_reason = "MODEL_MISSING"
                events.append({
                    "frame_id": frame_id,
                    "timestamp_sec": frame_id / info.fps if info.fps > 0 else 0.0,
                    "camera_id": video_path.stem.split("_")[0],
                    "track_id": int(track.get("track_id", -1)),
                    "face_detected": face_detected,
                    "face_bbox": local_face_bbox,
                    "embedding_dim": embedding_dim,
                    "embedding": embedding,
                    "face_quality": face_quality,
                    "spoof_status": spoof_status,
                    "failure_reason": final_reason,
                })
            frame_id += 1
    finally:
        capture.release()

    metrics = build_face_metrics(events, timer.elapsed(), str(events_path))
    metrics.update({
        "model_info": {"detector": config.get("detector"), "recognizer": config.get("recognizer"), "run_every_n_frames": run_every_n_frames},
        "input_video": str(video_path),
        "camera_id": video_path.stem.split("_")[0],
        "start_time": None,
        "total_frames": info.frame_count,
        "processed_frames": len({event["frame_id"] for event in events}),
        "output_paths": {"json": str(events_path), "metrics": str(metric_path)},
    })
    save_json(events_path, events)
    save_json(metric_path, metrics)
    print_saved(None, events_path, metric_path)
    return metrics


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    track_dir: str | Path | None = None,
    model_config: dict | None = None,
    run_every_n_frames: int = 10,
) -> list[dict]:
    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)
    print_header("CPose Face Module (Optional)")
    print_metric_table({
        "Input folder": resolve_path(input_dir),
        "Output folder": output_dir,
        "Track dir": resolve_path(track_dir) if track_dir else "(not provided)",
        "Videos found": len(videos),
    })
    results: list[dict] = []
    for video in videos:
        try:
            results.append(process_video(video, output_dir, track_dir, model_config, run_every_n_frames))
        except Exception as exc:
            print(f"ERROR processing face for {video.name}: {exc}")
            err_dir = ensure_dir(output_dir / video.stem)
            save_json(err_dir / "error.json", {"failure_reason": "STEP_FAILED", "error": str(exc)})
    return results
