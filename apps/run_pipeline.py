"""
apps/run_pipeline.py - CPose Full Pipeline (Body-only ReID)

Luồng mỗi frame:
  1. YOLO11-Pose → Bbox + 17 Keypoints
  2. ByteTrack   → local_track_id (ổn định, không nhảy)
  3. OSNet body ReID -> global_id (body_only realtime mode)
  5. PoseSequenceBuffer → khi đủ frame → EfficientGCN → action_label
  6. Metrics + UI log

Không hard-code đường dẫn tuyệt đối. Dùng --source để truyền vào.
"""

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
from src.core.event import EventBus, NullEventBus
from src.detectors.yolo_pose import YoloPoseTracker
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import get_module_source, load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import (
    create_video_writer,
    destroy_all_windows,
    find_default_video_source,
    get_video_meta,
    open_video_source,
    safe_imshow,
    toggle_video_recording,
)
from src.utils.vis import (
    FPSCounter,
    draw_adl_status,
    draw_detection,
    draw_info_panel,
    draw_reid_panel,
)

logger = get_logger(__name__)

WINDOW_NAME = "CPose - Full Pipeline"

# Optional label map for ADL (list indexed by label id)
ADL_LABEL_MAP = None


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CPose Full Pipeline - YOLO+ByteTrack+OSNet+EfficientGCN")
    p.add_argument("--source",      type=str, default=None)
    p.add_argument("--camera-id",   type=str, default="cam01")
    p.add_argument("--config",      type=str, default=str(ROOT / "configs/system/pipeline.yaml"))
    p.add_argument("--show",        action="store_true")
    p.add_argument("--no-show",     action="store_true")
    p.add_argument("--save-video",  action="store_true")
    p.add_argument("--save-events", action="store_true")
    p.add_argument("--save-clips",  action="store_true")
    p.add_argument("--output",      type=str, default=None)
    p.add_argument("--max-frames",  type=int, default=0)
    p.add_argument("--ui-log",      action="store_true", help="Gửi metrics tới UILogger")
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def clip_bbox(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame[y1:y2, x1:x2], [x1, y1, x2, y2]


def adl_label(status: dict | None) -> str:
    if not status:
        return "ADL: waiting"
    s = status.get("status", "waiting")
    if s == "collecting":
        return f"ADL: collecting {status.get('current_len', 0)}/{status.get('seq_len', 0)}"
    if s == "exported":
        return "ADL: exported"
    if s == "inferred":
        # If a label map is available, show human-readable name
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
    if s == "disabled":
        return "ADL: disabled"
    if s == "failed":
        return "ADL: failed"
    if s == "skipped":
        return f"ADL: skipped {status.get('reason', '')}".strip()
    return f"ADL: {s}"


def update_rule_window(rule_windows, track_id, keypoints, keypoint_scores, maxlen):
    if keypoints is None:
        return None
    window = rule_windows.setdefault(
        int(track_id),
        {"keypoints": deque(maxlen=maxlen), "scores": deque(maxlen=maxlen)},
    )
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


def build_overlay_label(tid, gid, score, weights, adl_status) -> str:
    mode = weights.get("mode", "")
    mode_short = {"face_dominant": "F↑", "body_dominant": "B↑", "balanced": "FB",
                  "face_only": "F", "body_only": "B", "no_modal": "??"}.get(mode, mode[:4])
    identity = (
        f"ID={gid}"
        if gid and gid not in {"unknown", "too_small"} and not str(gid).startswith("track_")
        else f"track={tid} UNK"
    )
    return (
        f"{identity} "
        f"r={score:.2f}[{mode_short}] "
        f"{adl_label(adl_status)}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def draw_object(frame, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    device = cfg["system"]["device"]

    # Load optional ADL label map (text file with one class name per line)
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

    # ── Detector + Tracker ───────────────────────────────────────────────────
    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        fallback_weights=cfg["pose"].get("fallback_weights"),
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracking"]["tracker_yaml"],
        device=device,
        tracking_cfg=cfg["tracking"],
    )
    tracker = ByteTrackWrapper(detector)

    # ── Body Extractor ───────────────────────────────────────────────────────
    osnet_reid = None
    reid_warning = None

    try:
        from src.reid.osnet_reid import OSNetReID

        reid_weights = cfg["reid"]["weights"]
        if not Path(reid_weights).exists() and cfg["reid"].get("fallback_weights"):
            reid_weights = cfg["reid"]["fallback_weights"]
        osnet_reid = OSNetReID(
            weight_path=reid_weights,
            threshold=cfg["reid"].get("threshold", 0.65),
            reid_interval=cfg["reid"].get("reid_interval", 15),
            max_gallery=cfg["reid"].get("max_gallery", 10),
            min_crop_area=cfg["reid"].get("min_crop_area", 2500),
            min_gallery_size=cfg["reid"].get("min_gallery_size", 1),
        )
        gallery_sources = (
            cfg["reid"].get("body_embedding_pkls")
            or cfg["reid"].get("body_embedding_dirs")
            or cfg["reid"].get("embedding_dirs")
        )
        loaded = osnet_reid.load_gallery_embeddings(gallery_sources, cfg["reid"].get("id_aliases"))
        if loaded <= 0:
            reid_warning = "OSNet gallery empty"
        else:
            logger.info(f"OSNet gallery loaded: {loaded} persons")

    except Exception as exc:
        reid_warning = f"OSNet ReID unavailable: {exc}"
        logger.warning(reid_warning)

    # ── ADL buffer + EfficientGCN ────────────────────────────────────────────
    adl_weights = cfg["adl"].get("weights", "models/2015_EfficientGCN-B0_ntu-xsub120.pth.tar")
    if not Path(adl_weights).exists() and cfg["adl"].get("fallback_weights"):
        adl_weights = cfg["adl"]["fallback_weights"]
    adl_model = EfficientGCNADL(
        weight_path=adl_weights,
        window=cfg["adl"].get("min_frames", 30),
        stride=cfg["adl"].get("infer_every_n_frames", 15),
        device="cpu",
    )
    if adl_model.load_error:
        logger.warning(f"EfficientGCN fallback=unknown: {adl_model.load_error}")

    object_model = None
    object_warning = None
    obj_cfg = cfg.get("object", {})
    if obj_cfg.get("enabled", False):
        try:
            from ultralytics import YOLO

            obj_weights = Path(obj_cfg.get("weights", ""))
            if not obj_weights.is_absolute():
                obj_weights = ROOT / obj_weights
            if not obj_weights.exists():
                raise FileNotFoundError(f"Object detector weights not found: {obj_weights}")
            object_model = YOLO(str(obj_weights))
            logger.info(f"Object detector loaded: {obj_weights}")
        except Exception as exc:
            object_warning = f"Object detector unavailable: {exc}"
            logger.warning(object_warning)

    # ── Event bus ────────────────────────────────────────────────────────────
    save_events = bool(args.save_events)
    event_bus = (
        EventBus(cfg["system"].get("event_log"), enabled=True)
        if save_events else NullEventBus()
    )

    # ── UILogger (optional) ──────────────────────────────────────────────────
    ui_logger = None
    if args.ui_log:
        try:
            from src.core.ui_logger import UILogger
            ui_logger = UILogger(max_lines=cfg.get("ui", {}).get("max_log_lines", 300))
        except Exception as exc:
            logger.warning(f"UILogger unavailable: {exc}")

    # ── Video source ─────────────────────────────────────────────────────────
    source = args.source or get_module_source(cfg, "pipeline") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError(
            "No video source. Set sources.pipeline or pass --source."
        )

    logger.info(f"Opening video source: {source}")
    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    panel_w = 220

    out_path = Path(args.output) if args.output else resolve_output_path(
        cfg["system"]["vis_dir"],
        make_video_output_name("pipe", args.camera_id),
    )
    writer = None
    if args.save_video:
        out_path = (
            Path(args.output) if args.output
            else resolve_output_path(
                cfg["system"]["vis_dir"],
                make_video_output_name("pipe", args.camera_id),
            )
        )
        writer = create_video_writer(out_path, fps, width + panel_w, height)
        logger.info(f"Saving video → {out_path}")

    # ── State ────────────────────────────────────────────────────────────────
    fps_counter = FPSCounter()
    track_status: dict[int, dict] = {}
    rule_windows: dict[int, dict] = {}
    prev_track_ids: set[int] = set()
    last_query_crop = None
    last_matches: list = []
    frame_idx = -1
    metrics_interval = int(cfg.get("ui", {}).get("metrics_interval_frames", 5))

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            h, w = frame.shape[:2]

            # ── 1+2: Detect + Track ──────────────────────────────────────────
            try:
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracking failed: {exc}", exc_info=True)
                continue

            object_count = 0
            if object_model is not None:
                try:
                    obj_results = object_model.predict(
                        source=frame,
                        conf=float(obj_cfg.get("conf", 0.45)),
                        iou=float(obj_cfg.get("iou", 0.5)),
                        device=device,
                        verbose=False,
                    )
                    obj_result = obj_results[0]
                    if obj_result.boxes is not None:
                        for box in obj_result.boxes:
                            cls_id = int(box.cls.item())
                            name = obj_result.names.get(cls_id, str(cls_id))
                            score = float(box.conf.item())
                            object_count += 1
                            draw_object(
                                frame,
                                box.xyxy[0].cpu().numpy().tolist(),
                                f"{name} {score:.2f}",
                            )
                except Exception as exc:
                    logger.warning(f"[frame {frame_idx}] object detection failed: {exc}", exc_info=True)

            curr_ids = {int(d.get("track_id", -1)) for d in detections if d.get("track_id", -1) >= 0}
            lost_ids = prev_track_ids - curr_ids

            # Cleanup lost tracks
            for tid in lost_ids:
                adl_model.cleanup_track(tid)
                rule_windows.pop(tid, None)
                if ui_logger:
                    ui_logger.log(args.camera_id, "INFO", "pipeline", f"Lost track id={tid}")
            for tid in lost_ids:
                track_status.pop(tid, None)
            prev_track_ids = curr_ids

            # ── 3+4: Body+Face ReID → GlobalID ──────────────────────────────
            for det in detections:
                tid = int(det.get("track_id", -1))
                crop, bbox_clipped = clip_bbox(frame, det["bbox"])

                # ── Face embedding (placeholder — connect ArcFace khi có) ───
                # No ArcFace branch is active in realtime yet.
                # body_only mode: ArcFace is not implemented in realtime yet.

                # ── Assign GlobalID ──────────────────────────────────────────
                if osnet_reid is not None and tid >= 0 and crop is not None:
                    try:
                        x1, y1, x2, y2 = map(float, det["bbox"])
                        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        gid, reid_score = osnet_reid.identify(crop, area)
                        reid_status = "gallery_match" if gid != "unknown" else "gallery_miss"
                        weights = {"mode": "body_only"}
                        last_matches = osnet_reid.get_top_matches(crop, topk=3)
                        last_query_crop = crop.copy()
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID failed tid={tid}: {exc}", exc_info=True)
                        gid, reid_score, reid_status, weights = f"track_{tid}", 0.0, "reid_failed", {}

                    # Top-K matches cho panel (nếu gallery có data)
                else:
                    gid = f"track_{tid}" if tid >= 0 else "unknown"
                    reid_score, reid_status, weights = 0.0, "reid_unavailable", {}
                    if crop is not None:
                        last_query_crop = crop.copy()

                # ── 5: ADL buffer ────────────────────────────────────────────
                adl_status = {"status": "waiting", "current_len": 0,
                              "seq_len": adl_model.window, "pkl_path": None}
                if tid >= 0:
                    rule_window = update_rule_window(
                        rule_windows,
                        tid,
                        det.get("keypoints"),
                        det.get("keypoint_scores"),
                        maxlen=adl_model.window,
                    )
                    label_name, action_score = adl_model.update(tid, det.get("keypoints"), frame_idx)
                    method = "efficientgcn"
                    if label_name == "unknown":
                        rule_status = infer_rule_status(rule_window, min_len=min(10, adl_model.window))
                        if rule_status and rule_status.get("status") == "inferred":
                            label_name = rule_status.get("label", "unknown")
                            action_score = float(rule_status.get("score", 0.0))
                            method = rule_status.get("method", "rule")
                    adl_status = {
                        "status": "inferred" if label_name != "unknown" else "collecting",
                        "label": label_name,
                        "score": float(action_score),
                        "method": method if label_name != "unknown" else None,
                        "current_len": len(adl_model.buffers.get(tid, [])),
                        "seq_len": adl_model.window,
                        "pkl_path": None,
                    }

                    if adl_status and adl_status.get("status") == "collecting":
                        previous = track_status.get(tid)
                        if previous and previous.get("status") == "inferred":
                            adl_status = {
                                **previous,
                                "current_len": adl_status.get("current_len"),
                                "seq_len": adl_status.get("seq_len"),
                            }

                track_status[tid] = adl_status

                # ── Draw per-track overlay ───────────────────────────────────
                label = build_overlay_label(tid, gid, reid_score, weights, adl_status)
                draw_detection(frame, det, label=label)

                # Event log
                event_bus.emit("track_update", {
                    "camera_id": args.camera_id, "frame_idx": frame_idx,
                    "local_track_id": tid, "global_id": gid,
                    "reid_score": float(reid_score), "reid_status": reid_status,
                    "fusion_mode": weights.get("mode", ""),
                    "bbox": bbox_clipped or list(map(int, det["bbox"])),
                    "adl_status": adl_status.get("status") if adl_status else "waiting",
                })

            # ── UILogger metrics (mỗi N frame) ───────────────────────────────
            current_fps = fps_counter.tick()
            first_status = next(iter(track_status.values()), {
                "status": "waiting", "current_len": 0, "seq_len": adl_model.window,
            })
            log_frame_metrics(
                logger,
                "pipeline",
                args.camera_id,
                frame_idx,
                current_fps,
                interval=metrics_interval,
                persons=len(detections),
                tracked=len(curr_ids),
                objects=object_count,
                reid_available=osnet_reid is not None,
                adl_status=first_status.get("status", "waiting"),
                action_label=first_status.get("label", "unknown"),
                action_score=f"{float(first_status.get('score', 0.0)):.2f}",
                action_method=first_status.get("method", ""),
                sequence_len=first_status.get("current_len", 0),
                seq_len_required=first_status.get("seq_len", adl_model.window),
            )
            if ui_logger and frame_idx % metrics_interval == 0:
                ui_logger.metric(args.camera_id, {
                    "camera_id": args.camera_id,
                    "module": "pipeline",
                    "frame_idx": frame_idx,
                    "fps": round(current_fps, 1),
                    "device": device,
                    "persons": len(detections),
                    "tracked": len(curr_ids),
                    "objects": object_count,
                    "reid_assigned": sum(
                        1 for s in track_status.values()
                        if s is not None
                    ) if osnet_reid else 0,
                    "adl_collecting": sum(
                        1 for s in track_status.values()
                        if s and s.get("status") == "collecting"
                    ),
                    "adl_exported": sum(
                        1 for s in track_status.values()
                        if s and s.get("status") in ("exported", "inferred")
                    ),
                    "status": "running",
                })

            # ── Info panel ───────────────────────────────────────────────────
            gallery_count = len(osnet_reid.gallery) if osnet_reid else 0
            info: dict = {
                "Module": "Full CPose (OSNet+EfficientGCN)",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Persons": len(detections),
                "Objects": object_count,
                "Gallery": gallery_count,
                "Device": device,
                "FPS": f"{current_fps:.1f}",
            }
            if reid_warning:
                info["Warning"] = reid_warning[:48]
            if object_warning:
                info["Object"] = object_warning[:48]

            draw_info_panel(frame, info)

            first_status = next(iter(track_status.values()), {
                "status": "waiting", "current_len": 0, "seq_len": adl_model.window,
            })
            draw_adl_status(frame, first_status, pos=(10, 170))
            display = draw_reid_panel(frame, last_query_crop, last_matches, panel_w=panel_w)

            # ── Output ───────────────────────────────────────────────────────
            if writer is not None:
                writer.write(display)
            if show:
                key = safe_imshow(WINDOW_NAME, display)
                if key in (ord("g"), ord("G")):
                    writer = toggle_video_recording(writer, out_path, fps, width + panel_w, height, logger)
                if key in (27, ord("q"), ord("Q")):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        destroy_all_windows()
        logger.info(f"Pipeline finished. Processed {frame_idx + 1} frames.")


if __name__ == "__main__":
    main()
