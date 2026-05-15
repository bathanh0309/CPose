import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow, toggle_video_recording
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11-Pose skeleton visualization")
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
    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracking"]["tracker_yaml"],
        device=cfg["system"]["device"],
        tracking_cfg=cfg["tracking"],
    )

    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    logger.info(f"Opening video source: {source}")
    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    out_path = Path(args.output) if args.output else resolve_output_path(
        cfg["system"]["vis_dir"],
        make_video_output_name("pose", args.camera_id),
    )
    writer = None
    if args.save_video:
        writer = create_video_writer(out_path, fps, width, height)
        logger.info(f"Recording started: {out_path}")

    if not cfg.get("output", {}).get("save_json", False):
        logger.info("Pose JSON disabled by config")

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
                detections, _ = detector.infer(frame, persist=None)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] pose inference failed: {exc}", exc_info=True)
                continue

            for det in detections:
                tid = det.get("track_id", -1)
                draw_detection(frame, det, label=f"track={tid} conf={det['score']:.2f}")

            keypoint_scores = [
                float(sum(det.get("keypoint_scores") or []) / len(det.get("keypoint_scores") or [1]))
                for det in detections
            ]
            avg_kpt = sum(keypoint_scores) / len(keypoint_scores) if keypoint_scores else 0.0
            valid_skeletons = sum(1 for det in detections if det.get("keypoints") is not None)
            fps_value = fps_counter.tick()
            draw_info_panel(frame, {
                "Module": "YOLO11-Pose",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Persons": len(detections),
                "Valid skeletons": valid_skeletons,
                "Avg kpt": f"{avg_kpt:.2f}",
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_value:.1f}",
            })
            log_frame_metrics(
                logger,
                "pose",
                args.camera_id,
                frame_idx,
                fps_value,
                interval=int(cfg.get("ui", {}).get("metrics_interval_frames", 5)),
                persons=len(detections),
                valid_skeletons=valid_skeletons,
                invalid_skeletons=max(0, len(detections) - valid_skeletons),
                avg_keypoint_score=f"{avg_kpt:.2f}",
            )

            if writer is not None:
                writer.write(frame)
            if show:
                key = safe_imshow("CPose - Pose Estimation", frame)
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
