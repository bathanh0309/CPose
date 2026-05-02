"""Live combined CPose pipeline.

This entrypoint is for demos: every frame is processed through detection,
tracking, pose, ADL, and ReID, then shown as one combined overlay.
"""
from __future__ import annotations

import argparse
import signal
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from src import DATA_TEST_DIR, OUTPUT_DIR, print_module_console
from src.common.config import get_section, load_model_registry, resolve_model_path
from src.common.json_io import save_json
from src.common.manifest import parse_video_timestamp_from_filename
from src.common.paths import ensure_dir, resolve_path
from src.common.timer import Timer
from src.common.topology import load_camera_topology
from src.common.video_io import create_video_writer, get_video_info, list_video_files, open_video, show_frame_preview
from src.common.visualization import draw_skeleton
from src.modules.adl_recognition.rule_based_adl import classify_adl, history_item
from src.modules.adl_recognition.schemas import ADLConfig, adl_config_from_dict
from src.modules.adl_recognition.smoothing import majority_vote
from src.modules.detection.detector import PersonDetector, resolve_detection_model
from src.modules.global_reid.global_id_manager import GlobalPersonTable
from src.modules.pose_estimation.api import _assign_track_ids
from src.modules.pose_estimation.pose_model import PoseModel, resolve_pose_model
from src.modules.tracking.tracker import SimpleIoUTracker


_STOP_REQUESTED = False


def _install_stop_handler() -> None:
    def _handler(_sig: int, _frame: Any) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print("\n[INFO] Stop requested; finishing current frame and closing live pipeline.")

    signal.signal(signal.SIGINT, _handler)


def _safe_run_tag(run_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_id or "live")).strip("_") or "live"


def _model_arg(resolved: Any) -> str | None:
    return str(resolved.path) if getattr(resolved, "path", None) is not None else None


def _draw_label(frame: Any, bbox: list[float], text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y1 = max(0, y1 - text_h - baseline - 10)
    label_y2 = label_y1 + text_h + baseline + 10
    label_x2 = min(frame.shape[1] - 1, x1 + text_w + 14)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(frame, text, (x1 + 7, label_y2 - baseline - 4), font, scale, (255, 255, 255), thickness)


def _draw_hud(frame: Any, active_rows: list[tuple[str, str]]) -> None:
    panel_w = 330
    panel_h = 92 + max(0, len(active_rows) - 1) * 26
    x1 = max(0, frame.shape[1] - panel_w - 14)
    y1 = 12
    x2 = frame.shape[1] - 14
    y2 = min(frame.shape[0] - 1, y1 + panel_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0, frame)
    cv2.putText(frame, "POSE + ADL + ReID", (x1 + 14, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 230, 70), 2)
    cv2.putText(frame, "Global IDs:", (x1 + 14, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1)
    for index, (gid, adl_label) in enumerate(active_rows[:5]):
        y = y1 + 86 + index * 26
        cv2.circle(frame, (x1 + 25, y - 6), 8, (0, 230, 70), -1)
        cv2.putText(frame, f"{gid}  {adl_label}", (x1 + 43, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)


def _combined_record(
    frame_id: int,
    timestamp_sec: float,
    camera_id: str,
    detections: list[dict],
    tracks: list[dict],
    persons: list[dict],
    reid_rows: list[dict],
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "camera_id": camera_id,
        "detections": detections,
        "tracks": tracks,
        "persons": persons,
        "reid": reid_rows,
        "failure_reason": "OK",
    }


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    detector: PersonDetector,
    pose_model: PoseModel,
    global_table: GlobalPersonTable,
    topology: Any,
    adl_config: ADLConfig,
    reid_config: dict[str, Any],
    track_iou_threshold: float = 0.30,
    min_hits: int = 1,
    max_age: int = 30,
    preview: bool = True,
) -> dict[str, Any]:
    global _STOP_REQUESTED
    video_path = resolve_path(video_path)
    camera_id, start_time = parse_video_timestamp_from_filename(video_path)
    video_out = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_out / "live_combined_overlay.mp4"
    records_path = video_out / "live_combined_records.json"
    metrics_path = video_out / "live_combined_metrics.json"

    info = get_video_info(video_path)
    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)
    tracker = SimpleIoUTracker(min_hits=min_hits, max_missing=max_age, window_size=adl_config.window_size)
    histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=adl_config.window_size))
    label_histories: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=adl_config.smoothing_frames))
    records: list[dict[str, Any]] = []
    frame_id = 0
    timer = Timer()

    print(f"[INFO] Live combined: {video_path.name} | frames={info.frame_count} | fps={info.fps:.2f}")
    try:
        while not _STOP_REQUESTED:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_sec = frame_id / info.fps if info.fps > 0 else 0.0
            current_time = start_time + timedelta(seconds=timestamp_sec) if start_time else None

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            persons = pose_model.estimate(frame)
            _assign_track_ids(
                persons,
                tracks,
                threshold=track_iou_threshold,
                run_on_confirmed_tracks_only=False,
            )

            active_hud_rows: list[tuple[str, str]] = []
            reid_rows: list[dict[str, Any]] = []
            for person in persons:
                track_id = int(person.get("track_id") if person.get("track_id") is not None else -1)
                raw_label, confidence, failure_reason = classify_adl(person, histories[track_id], adl_config)
                histories[track_id].appendleft(history_item(person))
                label_histories[track_id].append(raw_label)
                adl_label = majority_vote(label_histories[track_id])

                gp, match_info = global_table.match_or_create(
                    bbox=person.get("bbox", [0, 0, 0, 0]),
                    frame=frame,
                    camera_id=camera_id,
                    current_time=current_time,
                    track_id=track_id,
                    adl_label=adl_label,
                    keypoints=person.get("keypoints"),
                    face_event=None,
                    topology=topology,
                    config=reid_config,
                )
                gid_label = f"G{gp.gid}"
                display_label = f"{gid_label} | T{track_id} | {adl_label.upper()}"
                _draw_label(frame, person["bbox"], display_label, (0, 220, 30))
                draw_skeleton(frame, person.get("keypoints", []))
                active_hud_rows.append((gid_label, adl_label.upper()))
                reid_rows.append({
                    "local_track_id": track_id,
                    "global_id": gp.global_id,
                    "adl_label": adl_label,
                    "adl_confidence": confidence,
                    "match_status": match_info.get("match_status"),
                    "failure_reason": failure_reason if failure_reason != "OK" else match_info.get("failure_reason", "OK"),
                })

            _draw_hud(frame, active_hud_rows)
            writer.write(frame)
            if preview and show_frame_preview(f"CPose Combined | {video_path.stem}", frame, info.fps):
                _STOP_REQUESTED = True

            records.append(_combined_record(frame_id, timestamp_sec, camera_id, detections, tracks, persons, reid_rows))
            frame_id += 1
    except KeyboardInterrupt:
        _STOP_REQUESTED = True
        print("\n[INFO] Stop requested by keyboard.")
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = timer.elapsed()
    metrics = {
        "metric_type": "proxy",
        "module": "live_combined_pipeline",
        "input_video": str(video_path),
        "camera_id": camera_id,
        "processed_frames": frame_id,
        "fps_processing": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_latency_ms_per_frame": round(elapsed / frame_id * 1000.0, 2) if frame_id else 0.0,
        "output_paths": {
            "overlay": str(overlay_path),
            "json": str(records_path),
            "metrics": str(metrics_path),
        },
        "failure_reason": "OK",
    }
    save_json(records_path, records)
    save_json(metrics_path, metrics)
    print(f"[INFO] Saved combined overlay: {overlay_path}")
    return metrics


def process_folder(
    input_dir: str | Path,
    output_root: str | Path,
    models: str | Path | None = None,
    topology: str | Path | None = None,
    run_id: str = "live_combined",
    det_conf: float | None = None,
    pose_conf: float | None = None,
    keypoint_conf: float | None = None,
    preview: bool = True,
) -> Path:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    _install_stop_handler()

    registry = load_model_registry(models)
    det_model = resolve_model_path(registry, "human_detection")
    pose_model_ref = resolve_model_path(registry, "pose_estimation")
    det_cfg = get_section(registry, "human_detection")
    pose_cfg = get_section(registry, "pose_estimation")
    track_cfg = get_section(registry, "human_tracking")
    adl_cfg = get_section(registry, "adl_recognition")
    reid_cfg = get_section(registry, "global_reid")

    output_base = ensure_dir(output_root)
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ensure_dir(output_base / f"{run_ts}_{_safe_run_tag(run_id)}")
    combined_dir = ensure_dir(run_dir / "06_live_combined")
    topology_obj = load_camera_topology(topology)
    detector = PersonDetector(resolve_detection_model(_model_arg(det_model)), det_conf if det_conf is not None else float(det_cfg.get("conf", det_model.conf or 0.5)))
    pose_estimator = PoseModel(
        resolve_pose_model(_model_arg(pose_model_ref)),
        pose_conf if pose_conf is not None else float(pose_cfg.get("conf", pose_model_ref.conf or 0.45)),
        keypoint_conf=keypoint_conf if keypoint_conf is not None else float(pose_cfg.get("keypoint_conf", 0.30)),
        min_visible_keypoints=int(adl_cfg.get("min_visible_keypoints", 8)),
    )
    adl_config = adl_config_from_dict(adl_cfg, window_size=int(adl_cfg.get("window_size", 30)))
    global_table = GlobalPersonTable()
    videos = list_video_files(input_dir)

    save_json(run_dir / "live_run_config.json", {
        "input": str(resolve_path(input_dir)),
        "output": str(run_dir),
        "models": str(models) if models else None,
        "topology": str(topology) if topology else None,
        "video_count": len(videos),
        "failure_reason": "OK",
    })

    print("=" * 60)
    print("CPose Live Combined Pipeline")
    print("=" * 60)
    print(f"Input  : {resolve_path(input_dir)}")
    print(f"Output : {combined_dir}")
    print(f"Videos : {len(videos)}")
    print("Overlay: Detection + Tracking + Pose + ADL + ReID")
    print("=" * 60)

    results: list[dict[str, Any]] = []
    for index, video in enumerate(videos, 1):
        if _STOP_REQUESTED:
            break
        print(f"\n[{index}/{len(videos)}] {video.name}")
        results.append(process_video(
            video,
            combined_dir,
            detector,
            pose_estimator,
            global_table,
            topology_obj,
            adl_config,
            reid_cfg,
            track_iou_threshold=float(pose_cfg.get("track_iou_threshold", 0.30)),
            min_hits=1,
            max_age=int(track_cfg.get("max_age", 30)),
            preview=preview,
        ))

    save_json(run_dir / "global_person_table.json", global_table.to_dict())
    save_json(run_dir / "live_combined_summary.json", {
        "metric_type": "proxy",
        "processed_videos": len(results),
        "processed_frames": sum(int(row.get("processed_frames", 0)) for row in results),
        "outputs": [row.get("output_paths", {}).get("overlay") for row in results],
        "failure_reason": "OK",
    })
    print(f"\n[INFO] Live combined run complete: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPose combined per-frame live pipeline")
    parser.add_argument("--input", default=str(DATA_TEST_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "live"))
    parser.add_argument("--models", default="configs/model_registry.demo_i5.yaml")
    parser.add_argument("--config", default=None, help="Accepted for batch compatibility; use --models for model registry.")
    parser.add_argument("--topology", default="configs/camera_topology.yaml")
    parser.add_argument("--run-id", default="live_combined")
    parser.add_argument("--det-conf", type=float, default=None)
    parser.add_argument("--pose-conf", type=float, default=None)
    parser.add_argument("--keypoint-conf", type=float, default=None)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    print_module_console("live_pipeline", args)
    try:
        process_folder(
            args.input,
            args.output,
            models=args.models or args.config,
            topology=args.topology,
            run_id=args.run_id,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            keypoint_conf=args.keypoint_conf,
            preview=not args.no_preview,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Live combined pipeline stopped.")


if __name__ == "__main__":
    main()
