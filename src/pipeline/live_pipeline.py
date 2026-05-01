"""src/pipeline/live_pipeline.py
================================================================
CPose Live Pipeline — Unified frame-by-frame processing
================================================================
Runs ALL modules on every frame and shows ONE combined preview
window with:
  • Bounding boxes from detection (white)
  • Local track IDs from tracking (orange)
  • Skeleton from pose estimation (cyan)
  • ADL label from rule-based classifier (blue)

Output: one combined MP4 overlay per input video.
Press Q or Esc in the preview window to close it
(processing continues and MP4 is still saved).
Ctrl+C in the terminal stops everything cleanly.
================================================================
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import DATA_TEST_DIR, OUTPUT_DIR, print_module_console
from src.adl_recognition.adl_rules import classify_adl, history_item
from src.adl_recognition.config import ADLConfig
from src.common.metrics import Timer, save_json
from src.common.paths import ensure_dir, resolve_path
from src.common.video_io import (
    create_video_writer,
    get_video_info,
    list_video_files,
    open_video,
)
from src.common.visualization import draw_adl_label, draw_skeleton, draw_track
from src.human_detection.config import resolve_detection_model
from src.human_detection.detector import PersonDetector
from src.human_tracking.tracker import SimpleIoUTracker
from src.pose_estimation.api import _assign_track_ids
from src.pose_estimation.config import resolve_pose_model
from src.pose_estimation.pose_model import PoseModel

# ──────────────────────────────────────────────────────────────────────────────
# Global quit flag (set by Q key OR Ctrl+C)
# ──────────────────────────────────────────────────────────────────────────────
_QUIT_REQUESTED: bool = False


def _install_ctrl_c_handler() -> None:
    """Make Ctrl+C in the terminal set _QUIT_REQUESTED instead of crashing."""
    def _handler(sig: int, frame: Any) -> None:
        global _QUIT_REQUESTED
        print("\n[INFO] Ctrl+C received — finishing current frame then stopping.")
        _QUIT_REQUESTED = True

    signal.signal(signal.SIGINT, _handler)


def _check_key(window_name: str) -> bool:
    """Show/refresh the OpenCV event loop and return True if quit was requested.

    waitKey(16) ≈ 60 fps cap on the preview; also pumps the Win32 message queue
    so the window actually responds to keyboard input.
    """
    global _QUIT_REQUESTED
    if _QUIT_REQUESTED:
        return True
    key = cv2.waitKey(16) & 0xFF
    if key in (ord("q"), ord("Q"), 27):  # 27 = Esc
        _QUIT_REQUESTED = True
    return _QUIT_REQUESTED


# ──────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_all(
    frame,
    tracks: list[dict],
    persons: list[dict],
    adl_events: dict[int, tuple[str, float | None]],
) -> None:
    """Draw all overlays on *frame* in-place."""
    # 1. Track boxes (orange) with local T-IDs
    for track in tracks:
        draw_track(frame, track["bbox"], track["track_id"], track["confidence"])

    # 2. Skeleton + ADL label for every person with a confirmed track
    for person in persons:
        draw_skeleton(frame, person.get("keypoints", []))
        track_id = person.get("track_id")
        if track_id is not None:
            label, conf = adl_events.get(track_id, ("unknown", None))
            draw_adl_label(frame, person["bbox"], track_id, label, conf)

    # 3. HUD — frame info
    cv2.putText(
        frame,
        "Press Q/Esc to close preview",
        (8, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-video processor
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    det_model_path: str | None = None,
    pose_model_path: str | None = None,
    det_conf: float = 0.5,
    pose_conf: float = 0.45,
    keypoint_conf: float = 0.30,
    adl_window: int = 30,
    preview: bool = True,
) -> dict:
    global _QUIT_REQUESTED

    video_path = resolve_path(video_path)
    video_out_dir = ensure_dir(resolve_path(output_dir) / video_path.stem)
    overlay_path = video_out_dir / "live_overlay.mp4"
    json_path = video_out_dir / "live_records.json"

    # ── Initialise models ────────────────────────────────────────────────────
    detector = PersonDetector(resolve_detection_model(det_model_path), det_conf)
    tracker = SimpleIoUTracker(min_hits=3, max_missing=30, window_size=adl_window)
    pose_model = PoseModel(
        resolve_pose_model(pose_model_path), pose_conf,
        keypoint_conf=keypoint_conf, min_visible_keypoints=6,
    )
    adl_config = ADLConfig(window_size=adl_window)

    # Per-track sliding windows for ADL
    histories: dict[int, deque] = defaultdict(
        lambda: deque(maxlen=adl_config.window_size)
    )
    label_histories: dict[int, deque[str]] = defaultdict(
        lambda: deque(maxlen=adl_config.smoothing_frames)
    )

    info = get_video_info(video_path)
    capture = open_video(video_path)
    writer = create_video_writer(overlay_path, info.fps, info.width, info.height)

    window_name = f"CPose Live | {video_path.stem}"
    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    records: list[dict] = []
    frame_id = 0
    timer = Timer()
    print(f"[INFO] Processing: {video_path.name}  ({info.frame_count} frames @ {info.fps:.1f} fps)")

    try:
        while not _QUIT_REQUESTED:
            ok, frame = capture.read()
            if not ok:
                break

            ts = frame_id / info.fps if info.fps > 0 else 0.0
            cam_id = video_path.stem.split("_")[0]

            # ── Module 1: Detection ──────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Module 2: Tracking ───────────────────────────────────────────
            tracks = tracker.update(detections)

            # ── Module 3: Pose estimation ────────────────────────────────────
            persons = pose_model.estimate(frame)
            _assign_track_ids(persons, tracks, threshold=0.3)

            # ── Module 4: ADL classification ─────────────────────────────────
            adl_events: dict[int, tuple[str, float | None]] = {}
            for person in persons:
                tid = person.get("track_id")
                if tid is None:
                    tid = -1
                tid = int(tid)
                raw_label, confidence, _ = classify_adl(
                    person, histories[tid], adl_config
                )
                histories[tid].appendleft(history_item(person))
                label_histories[tid].append(raw_label)
                # majority-vote smoothing
                counts: dict[str, int] = {}
                for lbl in label_histories[tid]:
                    counts[lbl] = counts.get(lbl, 0) + 1
                smoothed = sorted(counts.items(), key=lambda x: -x[1])[0][0]
                adl_events[tid] = (smoothed, confidence)

            # ── Draw all overlays ────────────────────────────────────────────
            _draw_all(frame, tracks, persons, adl_events)

            # ── Write frame to MP4 ───────────────────────────────────────────
            writer.write(frame)

            # ── Show preview ─────────────────────────────────────────────────
            if preview:
                cv2.imshow(window_name, frame)
                _check_key(window_name)

            # ── Record ───────────────────────────────────────────────────────
            records.append({
                "frame_id": frame_id,
                "timestamp_sec": ts,
                "camera_id": cam_id,
                "tracks": tracks,
                "persons": persons,
                "adl_events": {str(k): v[0] for k, v in adl_events.items()},
                "failure_reason": "OK",
            })
            frame_id += 1

    except Exception as exc:
        print(f"[ERROR] Frame {frame_id}: {exc}")
    finally:
        capture.release()
        writer.release()
        if preview:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    elapsed = timer.elapsed()
    fps_proc = frame_id / elapsed if elapsed > 0 else 0.0
    print(
        f"[INFO] Done: {frame_id} frames | "
        f"{fps_proc:.2f} fps processing | "
        f"Saved → {overlay_path}"
    )
    save_json(json_path, records)

    return {
        "video": str(video_path),
        "frames": frame_id,
        "fps_processing": fps_proc,
        "overlay": str(overlay_path),
        "records_json": str(json_path),
        "quit_requested": _QUIT_REQUESTED,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Folder processor
# ──────────────────────────────────────────────────────────────────────────────

def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    det_model: str | None = None,
    pose_model: str | None = None,
    det_conf: float = 0.5,
    pose_conf: float = 0.45,
    keypoint_conf: float = 0.30,
    adl_window: int = 30,
    preview: bool = True,
) -> list[dict]:
    global _QUIT_REQUESTED
    _QUIT_REQUESTED = False  # reset for each folder run

    _install_ctrl_c_handler()

    videos = list_video_files(input_dir)
    output_dir = ensure_dir(output_dir)

    print("=" * 60)
    print("  CPose Live Pipeline — Detection + Tracking + Pose + ADL")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Videos : {len(videos)}")
    print("=" * 60)

    if not videos:
        print("[WARN] No video files found.")
        return []

    results: list[dict] = []
    for idx, video in enumerate(videos, 1):
        if _QUIT_REQUESTED:
            print("[INFO] Quit requested — skipping remaining videos.")
            break
        print(f"\n[{idx}/{len(videos)}] {video.name}")
        result = process_video(
            video, output_dir,
            det_model, pose_model,
            det_conf, pose_conf, keypoint_conf,
            adl_window, preview,
        )
        results.append(result)

    print("\n" + "=" * 60)
    print(f"  Completed {len(results)}/{len(videos)} videos.")
    print("=" * 60)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPose Live Pipeline — unified frame-by-frame processing"
    )
    parser.add_argument("--input", default=str(DATA_TEST_DIR), help="Folder with input .mp4 files")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "live"), help="Output folder")
    parser.add_argument("--det-model", default=None, help="Override detection model path")
    parser.add_argument("--pose-model", default=None, help="Override pose model path")
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--pose-conf", type=float, default=0.45)
    parser.add_argument("--keypoint-conf", type=float, default=0.30)
    parser.add_argument("--adl-window", type=int, default=30, help="ADL sliding window size (frames)")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window")
    args = parser.parse_args()

    print_module_console("live_pipeline", args)
    process_folder(
        args.input,
        args.output,
        args.det_model,
        args.pose_model,
        args.det_conf,
        args.pose_conf,
        args.keypoint_conf,
        args.adl_window,
        preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
