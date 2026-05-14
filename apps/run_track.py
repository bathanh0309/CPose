import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_pipeline_cfg
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.vis import draw_detection

DEFAULT_SOURCE = ROOT / "data" / "input" / "cam1_2026-01-29_16-26-25.mp4"
OUTPUT_DIR = ROOT / "data" / "output"
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug YOLO pose + ByteTrack")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="image/video path or webcam index",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.detectors.yolo_pose import YoloPoseTracker
    from src.trackers.bytetrack import ByteTrackWrapper

    cfg = load_pipeline_cfg(ROOT / "configs" / "system" / "pipeline.yaml", ROOT)
    conf = cfg["pose"]["conf"] if args.conf is None else args.conf

    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=conf,
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"],
    )
    tracker = ByteTrackWrapper(detector)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {source}")

    vis_dir = ensure_dir(OUTPUT_DIR / "vis")
    writer = None
    frame_idx = -1
    all_results = []

    logger.info(f"Tracking started | source={source} | camera={args.camera_id}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            if writer is None and args.save_vis:
                out_path = str(Path(vis_dir) / f"{args.camera_id}_track_debug.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                logger.info(f"Video writer opened: {out_path}")

            detections, _ = tracker.update(frame)
            serializable = []
            for det in detections:
                track_id = int(det.get("track_id", -1))
                label = f"t{track_id}" if track_id >= 0 else "untracked"
                draw_detection(frame, det, label=label)
                serializable.append(det)

            all_results.append({
                "frame_index": frame_idx,
                "detections": serializable,
            })

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("CPose Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("ESC pressed; stopping.")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    out_json = OUTPUT_DIR / "track_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved track results to: {out_json}")


if __name__ == "__main__":
    main()
