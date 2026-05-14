import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.pose_buffer import PoseSequenceBuffer
from src.core.event import EventBus
from src.detectors.yolo_pose import YoloPoseTracker
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel, draw_reid_panel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full CPose pipeline")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/system/pipeline.yaml"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def clipped_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2], [x1, y1, x2, y2]


def adl_label(status):
    if not status:
        return "ADL: waiting"
    if status.get("status") == "collecting":
        return f"ADL: collecting {status.get('current_len', 0)}/{status.get('seq_len', 0)}"
    if status.get("status") == "exported":
        return "ADL: exported"
    if status.get("status") == "inferred":
        return f"ADL: {status.get('label')} score={float(status.get('score', 0.0)):.2f}"
    if status.get("status") == "disabled":
        return "ADL: inference disabled"
    if status.get("status") == "failed":
        return "ADL: inference failed"
    if status.get("status") == "skipped":
        return f"ADL: skipped {status.get('reason', '')}".strip()
    return f"ADL: {status.get('status', 'waiting')}"


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    detector = YoloPoseTracker(cfg["pose"]["weights"], cfg["pose"]["conf"], cfg["pose"]["iou"], cfg["tracker"]["tracker_yaml"], cfg["system"]["device"])
    tracker = ByteTrackWrapper(detector)

    extractor = None
    gallery = None
    gid_mgr = None
    reid_warning = None
    try:
        from src.core.global_id import GlobalIDManager
        from src.reid.fast_reid import FastReIDExtractor
        from src.reid.gallery import ReIDGallery

        extractor = FastReIDExtractor(cfg["reid"]["fastreid_root"], cfg["reid"]["config"], cfg["reid"]["weights"], cfg["system"]["device"])
        gallery = ReIDGallery(extractor, cfg["reid"]["gallery_dir"])
        gallery.build()
        gid_mgr = GlobalIDManager(gallery, threshold=cfg["reid"]["threshold"], reid_interval=cfg["reid"]["reid_interval"])
    except Exception as exc:
        reid_warning = f"FastReID unavailable: {exc}"
        logger.warning(reid_warning)

    pose_buffer = PoseSequenceBuffer(
        seq_len=cfg["adl"]["seq_len"],
        stride=cfg["adl"]["stride"],
        output_dir=cfg["adl"]["export_dir"],
        default_label=cfg["adl"].get("default_label", 0),
        max_idle_frames=cfg["adl"].get("max_idle_frames", 150),
    )
    event_bus = EventBus(cfg["system"]["event_log"])

    posec3d_runner = None
    if cfg["adl"].get("auto_infer", False):
        try:
            from src.action.posec3d import PoseC3DRunner
            posec3d_runner = PoseC3DRunner(
                mmaction_root=cfg["adl"]["mmaction_root"],
                base_config=cfg["adl"]["base_config"],
                checkpoint=cfg["adl"]["weights"],
                work_dir=cfg["adl"].get("work_dir", cfg["adl"]["export_dir"]),
            )
        except Exception as exc:
            logger.warning(f"PoseC3D disabled: {exc}")

    source = args.source or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    panel_w = 220
    writer = None
    if args.save_video:
        output = args.output or str(Path(cfg["system"]["vis_dir"]) / f"{args.camera_id}_pipeline.mp4")
        writer = create_video_writer(output, fps, width + panel_w, height)
        logger.info(f"Saving video to: {output}")

    fps_counter = FPSCounter()
    track_status = {}
    last_query_crop = None
    last_matches = []
    prev_track_ids = set()
    frame_idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            h, w = frame.shape[:2]
            try:
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracking failed: {exc}", exc_info=True)
                continue

            curr_track_ids = {int(det.get("track_id", -1)) for det in detections if det.get("track_id", -1) >= 0}
            lost_track_ids = prev_track_ids - curr_track_ids
            if gid_mgr is not None:
                for lost_tid in lost_track_ids:
                    gid_mgr.forget_track(args.camera_id, lost_tid)
            for lost_tid in lost_track_ids:
                track_status.pop(lost_tid, None)
            prev_track_ids = curr_track_ids

            for det in detections:
                tid = det.get("track_id", -1)
                crop_info = clipped_crop(frame, det["bbox"])
                crop = crop_info[0] if crop_info else None
                bbox = crop_info[1] if crop_info else list(map(int, det["bbox"]))
                if gid_mgr is not None and tid >= 0 and crop is not None:
                    try:
                        gid, reid_score, reid_status = gid_mgr.assign(args.camera_id, tid, crop, frame_idx)
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID failed for track {tid}: {exc}", exc_info=True)
                        gid, reid_score, reid_status = f"track_{tid}", 0.0, "reid_failed"
                    try:
                        feat = extractor.extract(crop)
                        last_matches = gallery.get_top_matches(feat, topk=3)
                        last_query_crop = crop.copy()
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID panel extraction failed: {exc}")
                        last_matches = []
                else:
                    gid, reid_score, reid_status = (f"track_{tid}" if tid >= 0 else "unknown"), 0.0, "reid_unavailable"
                    if crop is not None:
                        last_query_crop = crop.copy()

                if tid >= 0:
                    adl_status = pose_buffer.update(
                        args.camera_id,
                        tid,
                        gid,
                        frame_idx,
                        det.get("keypoints"),
                        det.get("keypoint_scores"),
                        (h, w),
                    )
                    if adl_status and adl_status.get("status") == "exported":
                        event_bus.emit("pose_clip_exported", {
                            "camera_id": args.camera_id,
                            "frame_idx": frame_idx,
                            "local_track_id": int(tid),
                            "global_id": gid,
                            "pkl_path": adl_status.get("pkl_path"),
                        })
                        if posec3d_runner is None:
                            adl_status = {**adl_status, "status": "disabled"}
                        else:
                            try:
                                result = posec3d_runner.run_test(adl_status["pkl_path"])
                                if isinstance(result, dict) and result.get("label") is not None:
                                    adl_status = {
                                        "status": "inferred",
                                        "label": result["label"],
                                        "score": result.get("score", 0.0),
                                        "pkl_path": adl_status.get("pkl_path"),
                                    }
                                    event_bus.emit("adl_result", {
                                        "camera_id": args.camera_id,
                                        "frame_idx": frame_idx,
                                        "local_track_id": int(tid),
                                        "global_id": gid,
                                        "label": result["label"],
                                        "score": float(result.get("score", 0.0)),
                                        "pkl_path": adl_status.get("pkl_path"),
                                    })
                                elif isinstance(result, dict):
                                    adl_status = {**result, "pkl_path": adl_status.get("pkl_path")}
                            except Exception as exc:
                                logger.warning(f"PoseC3D failed for {adl_status['pkl_path']}: {exc}", exc_info=True)
                                adl_status = {"status": "failed", "label": None, "score": 0.0, "pkl_path": adl_status.get("pkl_path")}
                    track_status[tid] = adl_status
                else:
                    adl_status = {"status": "waiting", "current_len": 0, "seq_len": cfg["adl"]["seq_len"], "pkl_path": None}

                draw_detection(frame, det, label=f"t={tid} gid={gid} r={reid_score:.2f} {adl_label(adl_status)}")
                event_bus.emit("track_update", {
                    "camera_id": args.camera_id,
                    "frame_idx": frame_idx,
                    "local_track_id": int(tid),
                    "global_id": gid,
                    "reid_score": float(reid_score),
                    "reid_status": reid_status,
                    "bbox": bbox,
                    "adl_status": adl_status.get("status") if adl_status else "waiting",
                })

            info = {
                "Module": "Full CPose",
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
            first_status = next(iter(track_status.values()), {"status": "waiting", "current_len": 0, "seq_len": cfg["adl"]["seq_len"]})
            draw_adl_status(frame, first_status, pos=(10, 170))
            display = draw_reid_panel(frame, last_query_crop, last_matches, panel_w=panel_w)

            if writer is not None:
                writer.write(display)
            if show:
                key = safe_imshow("CPose - Full Pipeline", display)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
