import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.pose_buffer import PoseSequenceBuffer
from src.action.rule_adl import classify_rule_adl
from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow, toggle_video_recording
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel

logger = get_logger(__name__)

# Optional ADL label map (list indexed by label id)
ADL_LABEL_MAP = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11-Pose + pose clip export + optional PoseC3D ADL")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/system/pipeline.yaml"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-clips", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ui-log", action="store_true")
    return parser.parse_args()


def adl_text(status):
    if not status:
        return "ADL: waiting"
    state = status.get("status")
    if state == "collecting":
        return f"ADL: collecting {status.get('current_len', 0)}/{status.get('seq_len', 0)}"
    if state == "exported":
        return "ADL: clip exported"
    if state == "inferred":
        label_val = status.get('label')
        label_display = label_val
        try:
            if ADL_LABEL_MAP is not None and label_val is not None:
                lid = int(label_val)
                if 0 <= lid < len(ADL_LABEL_MAP):
                    label_display = ADL_LABEL_MAP[lid]
        except Exception:
            pass
        method = status.get("method")
        suffix = f" ({method})" if method else ""
        return f"ADL: {label_display} score={float(status.get('score', 0.0)):.2f}{suffix}"
    if state == "skipped":
        return f"ADL: skipped {status.get('reason', '')}".strip()
    if state == "failed":
        return "ADL: inference failed"
    if state == "disabled":
        return "ADL: inference disabled"
    return f"ADL: {state}"


def rule_status(pose_buffer, camera_id, track_id):
    window = pose_buffer.latest_window(camera_id, track_id)
    return classify_rule_adl(window)


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    # Load optional ADL label map
    global ADL_LABEL_MAP
    ADL_LABEL_MAP = None
    label_map_path = cfg.get("adl", {}).get("label_map_path")
    if label_map_path:
        p = Path(label_map_path)
        if not p.is_absolute():
            p = ROOT / label_map_path
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    ADL_LABEL_MAP = [l.strip() for l in fh if l.strip()]
                logger.info(f"Loaded ADL label map: {p} ({len(ADL_LABEL_MAP)} labels)")
            except Exception as exc:
                logger.warning(f"Failed to load ADL label map {p}: {exc}")
        else:
            logger.info(f"ADL label map not found: {p}")
    detector = YoloPoseTracker(
        cfg["pose"]["weights"],
        cfg["pose"]["conf"],
        cfg["pose"]["iou"],
        cfg["tracking"]["tracker_yaml"],
        cfg["system"]["device"],
        tracking_cfg=cfg["tracking"],
    )
    pose_buffer = PoseSequenceBuffer(
        seq_len=cfg["adl"]["seq_len"],
        stride=cfg["adl"]["stride"],
        output_dir=cfg["adl"]["export_dir"],
        default_label=cfg["adl"].get("default_label", 0),
        max_idle_frames=cfg["adl"].get("max_idle_frames", 150),
        export_enabled=args.save_clips or cfg["adl"].get("auto_infer", False),
    )

    posec3d_runner = None
    if cfg["adl"].get("auto_infer", False):
        try:
            from src.action.posec3d import PoseC3DRunner
            posec3d_runner = PoseC3DRunner(
                config=cfg["adl"]["posec3d_config"],
                checkpoint=cfg["adl"]["weights"],
                work_dir=cfg["adl"].get("work_dir", cfg["adl"]["export_dir"]),
                num_classes=cfg["adl"].get("num_classes", 60),
                mmaction_root=cfg["adl"].get("mmaction_root"),
            )
        except Exception as exc:
            logger.warning(f"PoseC3D disabled: {exc}")

    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    logger.info(f"Opening video source: {source}")
    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    writer = None
    out_path = Path(args.output) if args.output else resolve_output_path(
        cfg["system"]["vis_dir"],
        make_video_output_name("adl", args.camera_id),
    )
    if args.save_video:
        writer = create_video_writer(out_path, fps, width, height)
        logger.info(f"Recording started: {out_path}")

    if not cfg.get("output", {}).get("save_json", False):
        logger.info("ADL JSON disabled by config")

    track_status = {}
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
            h, w = frame.shape[:2]

            try:
                detections, _ = detector.infer(frame, persist=True)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] pose inference failed: {exc}", exc_info=True)
                continue

            for det in detections:
                tid = det.get("track_id", -1)
                if tid < 0:
                    draw_detection(frame, det, label="track=-1 ADL: waiting")
                    continue
                status = pose_buffer.update(args.camera_id, tid, f"track_{tid}", frame_idx, det.get("keypoints"), det.get("keypoint_scores"), (h, w))
                if status and status.get("status") == "exported":
                    if posec3d_runner is None:
                        status = rule_status(pose_buffer, args.camera_id, tid)
                    else:
                        try:
                            result = posec3d_runner.run_test(status["pkl_path"])
                            if isinstance(result, dict) and result.get("label") is not None:
                                raw_label = result.get("label")
                                label_name = raw_label
                                label_id = None
                                try:
                                    if ADL_LABEL_MAP is not None and raw_label is not None:
                                        lid = int(raw_label)
                                        if 0 <= lid < len(ADL_LABEL_MAP):
                                            label_name = ADL_LABEL_MAP[lid]
                                            label_id = lid
                                except Exception:
                                    label_name = str(raw_label)
                                status = {"status": "inferred", "label": label_name, "label_id": label_id, "score": result.get("score", 0.0)}
                            elif isinstance(result, dict):
                                status = result
                        except Exception as exc:
                            logger.warning(f"PoseC3D failed for {status['pkl_path']}: {exc}", exc_info=True)
                            status = {"status": "failed", "label": None, "score": 0.0}
                        if status.get("status") != "inferred":
                            status = rule_status(pose_buffer, args.camera_id, tid)
                elif status and status.get("status") == "disabled" and status.get("reason") == "clip_export_disabled":
                    status = rule_status(pose_buffer, args.camera_id, tid)
                elif status and status.get("status") == "collecting":
                    previous = track_status.get(tid)
                    if previous and previous.get("status") == "inferred":
                        status = {
                            **previous,
                            "current_len": status.get("current_len"),
                            "seq_len": status.get("seq_len"),
                        }
                track_status[tid] = status
                draw_detection(frame, det, label=f"track={tid} {adl_text(status)}")

            first_status = next(iter(track_status.values()), {"status": "waiting", "current_len": 0, "seq_len": cfg["adl"]["seq_len"]})
            fps_value = fps_counter.tick()
            draw_info_panel(frame, {
                "Module": "YOLO+Pose Buffer+PoseC3D",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Persons": len(detections),
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_value:.1f}",
            })
            log_frame_metrics(
                logger,
                "adl",
                args.camera_id,
                frame_idx,
                fps_value,
                interval=int(cfg.get("ui", {}).get("metrics_interval_frames", 5)),
                persons=len(detections),
                adl_status=first_status.get("status", "waiting"),
                sequence_len=first_status.get("current_len", 0),
                seq_len_required=first_status.get("seq_len", cfg["adl"]["seq_len"]),
            )
            draw_adl_status(frame, first_status, pos=(10, 150))

            if writer is not None:
                writer.write(frame)
            if show:
                key = safe_imshow("CPose - ADL", frame)
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
