from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import cv2

from src.adl_recognition.adl_rules import classify_adl, history_item
from src.adl_recognition.config import ADLConfig, adl_config_from_dict
from src.adl_recognition.metrics import build_adl_metrics
from src.common.console import print_header, print_metric_table, print_saved
from src.common.metrics import Timer, load_json, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video, show_frame_preview
from src.common.visualization import draw_adl_label, draw_skeleton


def _pose_files(pose_dir: str | Path) -> list[Path]:
    root = resolve_path(pose_dir)
    if not root.exists() and root.name == "03_pose":
        fallback = root.parent / "3_pose"
        if fallback.exists():
            print(f"[WARN] {root} not found, using {fallback}.")
            root = fallback
    if not root.exists():
        return []
    return sorted(root.glob("*/keypoints.json"))


def _video_for(stem: str, video_dir: str | Path) -> Path | None:
    for video in list_video_files(video_dir):
        if video.stem == stem:
            return video
    return None


def _majority_vote(labels: deque[str]) -> str:
    if not labels:
        return "unknown"
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], label_order(item[0])))[0][0]


def label_order(label: str) -> int:
    return 1 if label == "unknown" else 0


def process_pose_file(
    pose_json: str | Path,
    output_dir: str | Path,
    video_dir: str | Path,
    window_size: int = 30,
    config: ADLConfig | dict | None = None,
    preview: bool = True,
) -> dict:
    pose_json = resolve_path(pose_json)
    video_stem = pose_json.parent.name
    video_output_dir = ensure_dir(resolve_path(output_dir) / video_stem)
    events_path = video_output_dir / "adl_events.json"
    metric_path = video_output_dir / "adl_metrics.json"
    overlay_path = video_output_dir / "adl_overlay.mp4"
    pose_records = load_json(pose_json, [])
    adl_config = config if isinstance(config, ADLConfig) else adl_config_from_dict(config, window_size=window_size)
    histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=adl_config.window_size))
    label_histories: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=adl_config.smoothing_frames))
    events: list[dict] = []
    timer = Timer()

    for record in pose_records:
        for person in record.get("persons", []):
            track_id = person.get("track_id")
            if track_id is None:
                track_id = -1
            track_id = int(track_id)
            raw_label, confidence, failure_reason = classify_adl(person, histories[track_id], adl_config)
            histories[track_id].appendleft(history_item(person))
            label_histories[track_id].append(raw_label)
            smoothed_label = _majority_vote(label_histories[track_id])
            events.append({
                "frame_id": int(record["frame_id"]),
                "timestamp_sec": float(record.get("timestamp_sec", 0.0)),
                "camera_id": record.get("camera_id", video_stem.split("_")[0]),
                "track_id": track_id,
                "raw_label": raw_label,
                "smoothed_label": smoothed_label,
                "adl_label": smoothed_label,
                "confidence": confidence,
                "window_size": adl_config.window_size,
                "visible_keypoint_ratio": person.get("visible_keypoint_ratio"),
                "failure_reason": failure_reason,
            })

    video_path = _video_for(video_stem, video_dir)
    saved_overlay: Path | None = None
    if video_path:
        info = get_video_info(video_path)
        capture = open_video(video_path)
        writer = create_video_writer(overlay_path, info.fps, info.width, info.height)
        event_by_frame_track = {(event["frame_id"], event["track_id"]): event for event in events}
        frame_id = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                persons = pose_records[frame_id].get("persons", []) if frame_id < len(pose_records) else []
                for person in persons:
                    track_id = int(person.get("track_id") if person.get("track_id") is not None else -1)
                    event = event_by_frame_track.get((frame_id, track_id))
                    if event:
                        draw_adl_label(frame, person["bbox"], track_id, event["adl_label"], event["confidence"])
                    draw_skeleton(frame, person.get("keypoints", []))
                writer.write(frame)
                if preview:
                    show_frame_preview(f"CPose ADL | {video_stem}", frame)
                frame_id += 1
        finally:
            capture.release()
            writer.release()
            cv2.destroyWindow(f"CPose ADL | {video_stem}")
        saved_overlay = overlay_path

    metrics = build_adl_metrics(events, timer.elapsed(), str(events_path), str(saved_overlay) if saved_overlay else None)
    metrics.update({
        "metric_type": "proxy",
        "model_info": {"method": "rule_based", "window_size": adl_config.window_size, "smoothing_frames": adl_config.smoothing_frames},
        "input_video": str(video_path) if video_path else None,
        "camera_id": video_stem.split("_")[0],
        "start_time": None,
        "output_paths": {"overlay": str(saved_overlay) if saved_overlay else None, "json": str(events_path), "metrics": str(metric_path)},
        "failure_reason": "OK",
    })
    save_json(events_path, events)
    save_json(metric_path, metrics)
    print(f"Frames: {len(pose_records)} | ADL events: {metrics['total_adl_events']} | FPS-eq: {metrics['fps_equivalent']:.2f}")
    print_saved(saved_overlay, events_path, metric_path)
    return metrics


def process_folder(
    pose_dir: str | Path,
    video_dir: str | Path,
    output_dir: str | Path,
    window_size: int = 30,
    config: ADLConfig | dict | None = None,
    preview: bool = True,
) -> list[dict]:
    files = _pose_files(pose_dir)
    output_dir = ensure_dir(output_dir)
    print_header("CPose ADL Recognition Module")
    print_metric_table({
        "Pose folder": resolve_path(pose_dir),
        "Video folder": resolve_path(video_dir),
        "Output folder": output_dir,
        "Window size": window_size,
        "Pose files found": len(files),
    })
    if not files:
        print("No pose keypoints found.")
        return []
    results = []
    for index, pose_file in enumerate(files, 1):
        print(f"\n[{index}/{len(files)}] Processing {pose_file.parent.name}")
        try:
            results.append(process_pose_file(pose_file, output_dir, video_dir, window_size, config, preview))
        except Exception as exc:
            print(f"ERROR processing {pose_file}: {exc}")
    print_header("SUMMARY")
    print_metric_table({
        "Total videos": len(results),
        "Total ADL events": sum(item.get("total_adl_events", 0) for item in results),
        "Average FPS equivalent": f"{sum(item.get('fps_equivalent', 0.0) for item in results) / len(results):.2f}" if results else "0.00",
    })
    return results
