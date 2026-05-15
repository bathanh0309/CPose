import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel, draw_reid_panel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11-Pose + ByteTrack + FastReID visualization")
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


def clipped_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    detector = YoloPoseTracker(
        cfg["pose"]["weights"],
        cfg["pose"]["conf"],
        cfg["pose"]["iou"],
        cfg["tracking"]["tracker_yaml"],
        cfg["system"]["device"],
        tracking_cfg=cfg["tracking"],
    )
    extractor = None
    gallery = None
    gid_mgr = None
    reid_warning = None
    try:
        from src.core.global_id import GlobalIDManager
        from src.reid.fast_reid import FastReIDExtractor
        from src.reid.gallery import ReIDGallery

        extractor = FastReIDExtractor(
            config=cfg["reid"]["fastreid_config"],
            weights_path=cfg["reid"]["weights"],
            device=cfg["system"]["device"],
            output_dir=cfg["reid"].get("output_dir"),
            fastreid_root=cfg["reid"].get("fastreid_root"),
        )
        gallery = ReIDGallery(extractor, cfg["reid"]["gallery_dir"], embedding_dirs=cfg["reid"].get("embedding_dirs"))
        gallery.build()
        gid_mgr = GlobalIDManager(gallery, threshold=cfg["reid"]["threshold"], reid_interval=cfg["reid"]["reid_interval"])
    except Exception as exc:
        reid_warning = f"FastReID unavailable: {exc}"
        logger.warning(reid_warning)

    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    show = bool(args.show and not args.no_show)
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    writer = None
    save_video = bool(args.save_video)
    if save_video:
        out_path = Path(args.output) if args.output else resolve_output_path(
            cfg["system"]["vis_dir"],
            make_video_output_name("reid", args.camera_id),
        )
        writer = create_video_writer(out_path, fps, width + 220, height)
        logger.info(f"Saving video to: {out_path}")

    if not cfg.get("output", {}).get("save_json", False):
        logger.info("ReID JSON disabled by config")

    last_crop = None
    last_matches = []
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
                detections, _ = detector.infer(frame, persist=True)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] pose tracking failed: {exc}", exc_info=True)
                continue

            for det in detections:
                tid = det.get("track_id", -1)
                crop = clipped_crop(frame, det["bbox"])
                if tid < 0 or crop is None:
                    draw_detection(frame, det, label=f"track={tid}")
                    continue
                if gid_mgr is not None:
                    gid, score, status = gid_mgr.assign(args.camera_id, tid, crop, frame_idx)
                    try:
                        feat = extractor.extract(crop)
                        last_matches = gallery.get_top_matches(feat, topk=3)
                        last_crop = crop.copy()
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID panel extraction failed: {exc}")
                        last_matches = []
                else:
                    gid, score, status = f"track_{tid}", 0.0, "reid_unavailable"
                    last_crop = crop.copy()
                    last_matches = []
                draw_detection(frame, det, label=f"track={tid} gid={gid} score={score:.2f} {status}")

            info = {
                "Module": "YOLO+ByteTrack+FastReID",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Persons": len(detections),
                "Gallery": len(gallery.prototypes) if gallery is not None else 0,
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_counter.tick():.1f}",
            }
            if reid_warning:
                info["Warning"] = reid_warning[:48]
            elif gallery is not None and gallery.initial_empty:
                info["Warning"] = "ReID gallery empty"
            draw_info_panel(frame, info)
            display = draw_reid_panel(frame, last_crop, last_matches, panel_w=220)

            if writer is not None:
                writer.write(display)
            if show:
                key = safe_imshow("CPose - ReID", display)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
