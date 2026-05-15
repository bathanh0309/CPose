import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.pedestrian_yolo import PedestrianYoloTracker
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel, track_color

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run custom pedestrian YOLO + ByteTrack")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/system/pipeline.yaml"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-trails", action="store_true")
    return parser.parse_args()


def draw_track_trails(frame, history):
    for track_id, points in history.items():
        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.int32)
        cv2.polylines(frame, [pts], False, track_color(track_id), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    ped_cfg = cfg.get("pedestrian", {})

    detector = PedestrianYoloTracker(
        weights=ped_cfg.get("weights", ROOT / "models/tracking.pt"),
        conf=ped_cfg.get("conf", 0.3),
        iou=ped_cfg.get("iou", 0.5),
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"],
        classes=ped_cfg.get("classes"),
    )

    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    writer = None
    save_video = bool(args.save_video)
    if save_video:
        out_path = Path(args.output) if args.output else resolve_output_path(
            cfg["system"]["vis_dir"],
            make_video_output_name("pedestrian", args.camera_id),
        )
        writer = create_video_writer(out_path, fps, width, height)
        logger.info(f"Saving video to: {out_path}")

    fps_counter = FPSCounter()
    track_history = defaultdict(list)
    frame_idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            try:
                detections, _ = detector.infer(frame, persist=True)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] pedestrian tracking failed: {exc}", exc_info=True)
                continue

            tracked = 0
            current_ids = set()
            for det in detections:
                tid = det.get("track_id", -1)
                if tid >= 0:
                    tracked += 1
                    current_ids.add(tid)
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    track_history[tid].append(center)
                    if len(track_history[tid]) > 30:
                        track_history[tid].pop(0)
                draw_detection(frame, det, label=f"ped={tid} conf={det['score']:.2f}")

            if not args.no_trails:
                draw_track_trails(frame, track_history)

            for stale_id in list(track_history.keys()):
                if stale_id not in current_ids and len(track_history[stale_id]) > 30:
                    track_history.pop(stale_id, None)

            draw_info_panel(frame, {
                "Module": "Custom Pedestrian YOLO + ByteTrack",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Detections": len(detections),
                "Tracked": tracked,
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_counter.tick():.1f}",
            })

            if writer is not None:
                writer.write(frame)
            if show:
                key = safe_imshow("CPose - Pedestrian Tracking", frame)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
