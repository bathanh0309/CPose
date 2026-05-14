import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.pose_buffer import PoseSequenceBuffer
from src.action.posec3d import PoseC3DRunner
from src.core.event import EventBus
from src.core.global_id import GlobalIDManager
from src.detectors.yolo_pose import YoloPoseTracker
from src.reid.fast_reid import FastReIDExtractor
from src.reid.gallery import ReIDGallery
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import load_pipeline_cfg
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.vis import draw_detection

logger = get_logger("pipeline")


def load_cfg(path: Path):
    return load_pipeline_cfg(path, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="CPose inference pipeline")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def build_pipeline(cfg):
    logger.info("Building detector...")
    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"]
    )
    tracker = ByteTrackWrapper(detector)

    logger.info("Building ReID extractor...")
    reid_extractor = FastReIDExtractor(
        fastreid_root=cfg["reid"]["fastreid_root"],
        config_path=cfg["reid"]["config"],
        weights_path=cfg["reid"]["weights"],
        device=cfg["system"]["device"]
    )

    gallery = ReIDGallery(reid_extractor, cfg["reid"]["gallery_dir"])
    gallery.build()

    global_id_manager = GlobalIDManager(
        gallery,
        threshold=cfg["reid"]["threshold"],
        reid_interval=cfg["reid"].get("reid_interval", 10),
    )

    pose_buffer = PoseSequenceBuffer(
        seq_len=cfg["adl"]["seq_len"],
        stride=cfg["adl"]["stride"],
        output_dir=cfg["adl"]["export_dir"],
        default_label=cfg["adl"]["default_label"],
        max_idle_frames=cfg["adl"].get("max_idle_frames", 150),
    )

    event_bus = EventBus(cfg["system"]["event_log"])

    posec3d_runner = None
    if cfg["adl"].get("auto_infer", False):
        logger.info("Building PoseC3D runner...")
        posec3d_runner = PoseC3DRunner(
            mmaction_root=cfg["adl"]["mmaction_root"],
            base_config=cfg["adl"]["base_config"],
            checkpoint=cfg["adl"]["weights"],
            work_dir=cfg["adl"]["work_dir"]
        )

    return tracker, global_id_manager, pose_buffer, event_bus, posec3d_runner


def main():
    args = parse_args()
    cfg = load_cfg(Path(args.config))

    tracker, global_id_manager, pose_buffer, event_bus, posec3d_runner = build_pipeline(cfg)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    vis_dir = ensure_dir(cfg["system"]["vis_dir"])
    writer = None
    frame_idx = -1

    logger.info(f"Pipeline started | source={source} | camera={args.camera_id}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("End of stream.")
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            if writer is None and cfg["system"].get("save_video", False):
                out_path = str(Path(vis_dir) / f"{args.camera_id}_tracked.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                logger.info(f"Video writer opened: {out_path}")

            try:
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracker.update failed: {exc}", exc_info=True)
                continue

            for det in detections:
                if det.get("track_id", -1) < 0:
                    continue

                x1, y1, x2, y2 = map(int, det["bbox"])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                try:
                    global_id, reid_score = global_id_manager.assign(
                        camera_id=args.camera_id,
                        local_track_id=det["track_id"],
                        crop_bgr=crop,
                        frame_idx=frame_idx,
                    )
                except Exception as exc:
                    logger.warning(f"ReID failed track {det['track_id']}: {exc}", exc_info=True)
                    global_id, reid_score = f"unk_{det['track_id']}", 0.0

                pkl_path = pose_buffer.update(
                    camera_id=args.camera_id,
                    local_track_id=det["track_id"],
                    global_id=global_id,
                    frame_idx=frame_idx,
                    keypoints_xy=det.get("keypoints"),
                    keypoint_scores=det.get("keypoint_scores"),
                    img_shape=(h, w)
                )

                label = f"{global_id} | t{det['track_id']} | {reid_score:.2f}"
                draw_detection(frame, det, label=label)

                event_bus.emit("track_update", {
                    "camera_id": args.camera_id,
                    "frame_idx": frame_idx,
                    "local_track_id": int(det["track_id"]),
                    "global_id": global_id,
                    "reid_score": float(reid_score),
                    "bbox": [x1, y1, x2, y2]
                })

                if pkl_path is not None:
                    event_bus.emit("pose_clip_exported", {
                        "camera_id": args.camera_id,
                        "frame_idx": frame_idx,
                        "local_track_id": int(det["track_id"]),
                        "global_id": global_id,
                        "pkl_path": str(pkl_path)
                    })

                    if posec3d_runner is not None:
                        try:
                            posec3d_runner.run_test(str(pkl_path))
                        except Exception as exc:
                            logger.warning(f"PoseC3D inference failed: {exc}", exc_info=True)

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("CPose Pipeline", frame)
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
        logger.info("Pipeline stopped cleanly.")


if __name__ == "__main__":
    main()
