import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.yolo_pose import YoloPoseTracker
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ByteTrack over YOLO11-Pose detections")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/system/pipeline.yaml"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)

    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"],
    )
    tracker = ByteTrackWrapper(detector)

    source = args.source or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    writer = None
    if args.save_video:
        output = args.output or str(Path(cfg["system"]["vis_dir"]) / f"{args.camera_id}_track.mp4")
        writer = create_video_writer(output, fps, width, height)
        logger.info(f"Saving video to: {output}")

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
            for det in detections:
                tid = det.get("track_id", -1)
                if tid >= 0:
                    tracked += 1
                draw_detection(frame, det, label=f"track={tid} conf={det['score']:.2f}")

            draw_info_panel(frame, {
                "Module": "YOLO11-Pose + ByteTrack",
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
                key = safe_imshow("CPose - Tracking", frame)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
