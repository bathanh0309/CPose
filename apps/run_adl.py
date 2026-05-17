import argparse
import sys
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.efficientgcn_adl import EfficientGCNADL
from src.action.rule_adl import classify_rule_adl
from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.config import get_module_source, load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, destroy_all_windows, find_default_video_source, get_video_meta, open_video_source, safe_imshow, toggle_video_recording
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel

logger = get_logger(__name__)

# Optional ADL label map (list indexed by label id)
ADL_LABEL_MAP = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11-Pose + EfficientGCN-B0 ADL")
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


def update_rule_window(rule_windows, track_id, keypoints, keypoint_scores, maxlen):
    if keypoints is None:
        return None
    window = rule_windows.setdefault(int(track_id), {"keypoints": deque(maxlen=maxlen), "scores": deque(maxlen=maxlen)})
    window["keypoints"].append(keypoints)
    if keypoint_scores is None:
        import numpy as np
        keypoint_scores = np.ones((len(keypoints),), dtype="float32")
    window["scores"].append(keypoint_scores)
    return window


def infer_rule_status(rule_window, min_len):
    if rule_window is None or len(rule_window["keypoints"]) < min_len:
        return None
    return classify_rule_adl(
        {
            "keypoints": list(rule_window["keypoints"]),
            "scores": list(rule_window["scores"]),
        }
    )


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
    adl_model = EfficientGCNADL(
        weight_path=cfg["adl"].get("weights", "models/2015_EfficientGCN-B0_ntu-xsub120.pth.tar"),
        window=cfg["adl"].get("min_frames", 30),
        stride=cfg["adl"].get("infer_every_n_frames", 15),
        device="cpu",
    )
    if adl_model.load_error:
        logger.warning(f"EfficientGCN fallback=unknown: {adl_model.load_error}")

    source = args.source or get_module_source(cfg, "adl") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Set sources.adl or pass --source.")

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
    rule_windows = {}
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
                rule_window = update_rule_window(
                    rule_windows,
                    tid,
                    det.get("keypoints"),
                    det.get("keypoint_scores"),
                    maxlen=adl_model.window,
                )
                label, score = adl_model.update(tid, det.get("keypoints"), frame_idx)
                current_len = len(adl_model.buffers.get(tid, []))
                method = "efficientgcn"
                if label == "unknown":
                    rule_status = infer_rule_status(rule_window, min_len=min(10, adl_model.window))
                    if rule_status and rule_status.get("status") == "inferred":
                        label = rule_status.get("label", "unknown")
                        score = float(rule_status.get("score", 0.0))
                        method = rule_status.get("method", "rule")
                status = {
                    "status": "inferred" if label != "unknown" else "collecting",
                    "label": label,
                    "score": score,
                    "method": method if label != "unknown" else None,
                    "current_len": current_len,
                    "seq_len": adl_model.window,
                }
                track_status[tid] = status
                draw_detection(frame, det, label=f"track={tid} {adl_text(status)}")

            first_status = next(iter(track_status.values()), {"status": "waiting", "current_len": 0, "seq_len": adl_model.window})
            fps_value = fps_counter.tick()
            draw_info_panel(frame, {
                "Module": "YOLO+EfficientGCN-B0",
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
                action_label=first_status.get("label", "unknown"),
                action_score=f"{float(first_status.get('score', 0.0)):.2f}",
                action_method=first_status.get("method", ""),
                sequence_len=first_status.get("current_len", 0),
                seq_len_required=first_status.get("seq_len", adl_model.window),
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
        destroy_all_windows()


if __name__ == "__main__":
    main()
