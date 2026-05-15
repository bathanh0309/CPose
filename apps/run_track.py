import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.yolo_pose import YoloPoseTracker
from src.detectors.pedestrian_yolo import PedestrianYoloTracker
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow, toggle_video_recording
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pedestrian detector + ByteTrack")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/system/pipeline.yaml"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ui-log", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)

    if cfg["tracking"].get("model_type") == "pedestrian":
        detector = PedestrianYoloTracker(
            weights=cfg["tracking"].get("weights", ROOT / "models/tracking.pt"),
            conf=cfg["tracking"].get("person_conf", 0.60),
            iou=cfg["tracking"].get("iou", 0.5),
            tracker=cfg["tracking"]["tracker_yaml"],
            device=cfg["system"]["device"],
            classes=cfg["tracking"].get("classes", [0]),
            tracking_cfg=cfg["tracking"],
        )
    else:
        detector = YoloPoseTracker(
            weights=cfg["pose"]["weights"],
            conf=cfg["pose"]["conf"],
            iou=cfg["pose"]["iou"],
            tracker=cfg["tracking"]["tracker_yaml"],
            device=cfg["system"]["device"],
            tracking_cfg=cfg["tracking"],
        )
    tracker = ByteTrackWrapper(detector)

    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    logger.info(f"Opening video source: {source}")
    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    out_path = Path(args.output) if args.output else resolve_output_path(
        cfg["system"]["vis_dir"],
        make_video_output_name("track", args.camera_id),
    )
    writer = None
    if args.save_video:
        writer = create_video_writer(out_path, fps, width, height)
        logger.info(f"Recording started: {out_path}")

    if not cfg.get("output", {}).get("save_json", False):
        logger.info("Track JSON disabled by config")

    fps_counter = FPSCounter()
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
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracking failed: {exc}", exc_info=True)
                continue

            tracked = 0
            filter_stats = detector.last_filter_stats.as_dict()
            for det in detections:
                tid = det.get("track_id", -1)
                if tid >= 0:
                    tracked += 1
                draw_detection(frame, det, label=f"track={tid} conf={det['score']:.2f}")

            fps_value = fps_counter.tick()
            info = {
                "Module": "Pedestrian Tracking",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Detections": len(detections),
                "Tracked": tracked,
                "Filtered": sum(filter_stats.values()),
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_value:.1f}",
            }
            log_frame_metrics(
                logger,
                "tracking",
                args.camera_id,
                frame_idx,
                fps_value,
                interval=int(cfg.get("ui", {}).get("metrics_interval_frames", 5)),
                detections=len(detections),
                tracked=tracked,
                **filter_stats,
            )
            if detector.last_filter_stats.warnings:
                info["Warning"] = detector.last_filter_stats.warnings[-1]
            draw_info_panel(frame, info)

            if writer is not None:
                writer.write(frame)
            if show:
                key = safe_imshow("CPose - Pedestrian Tracking", frame)
                if key in (ord("g"), ord("G")):
                    writer = toggle_video_recording(writer, out_path, fps, width, height, logger)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
