"""
apps/run_pipeline.py — CPose Full Pipeline (Multi-Modal ReID)

Luồng mỗi frame:
  1. YOLO11-Pose → Bbox + 17 Keypoints
  2. ByteTrack   → local_track_id (ổn định, không nhảy)
  3. (Parallel)  ArcFace/FaceDetector → face_feat + face_conf  (nếu thấy mặt)
                 BodyExtractor        → body_feat              (luôn trích xuất)
  4. GlobalIDManager (Face+Body Fusion) → global_id
  5. PoseSequenceBuffer → khi đủ 48 frame → PoseC3D → action_label
  6. Metrics + UI log

Không hard-code đường dẫn tuyệt đối. Dùng --source để truyền vào.
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.pose_buffer import PoseSequenceBuffer
from src.core.event import EventBus, NullEventBus
from src.detectors.yolo_pose import YoloPoseTracker
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import (
    create_video_writer,
    find_default_video_source,
    get_video_meta,
    open_video_source,
    safe_imshow,
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CPose Full Pipeline — Track+Pose+ReID(Face+Body)+ADL")
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
        return f"ADL: {status.get('label')} score={float(status.get('score', 0.0)):.2f}"
    if s == "disabled":
        return "ADL: disabled"
    if s == "failed":
        return "ADL: failed"
    if s == "skipped":
        return f"ADL: skipped {status.get('reason', '')}".strip()
    return f"ADL: {s}"


def build_overlay_label(tid, gid, score, weights, adl_status) -> str:
    mode = weights.get("mode", "")
    mode_short = {"face_dominant": "F↑", "body_dominant": "B↑", "balanced": "FB",
                  "face_only": "F", "body_only": "B", "no_modal": "??"}.get(mode, mode[:4])
    return (
        f"t={tid} gid={gid} "
        f"r={score:.2f}[{mode_short}] "
        f"{adl_label(adl_status)}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    device = cfg["system"]["device"]

    # ── Detector + Tracker ───────────────────────────────────────────────────
    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=device,
    )
    tracker = ByteTrackWrapper(detector)

    # ── Body Extractor ───────────────────────────────────────────────────────
    body_extractor = None
    multimodal_gallery = None
    gid_mgr = None
    reid_warning = None

    try:
        from src.reid.body_extractor import BodyExtractor
        from src.reid.fast_reid import FastReIDExtractor
        from src.reid.fusion import MultiModalGallery
        from src.core.global_id import GlobalIDManager

        extractor = FastReIDExtractor(
            config=cfg["reid"]["fastreid_config"],
            weights_path=cfg["reid"]["weights"],
            device=device,
            output_dir=cfg["reid"].get("output_dir"),
            fastreid_root=cfg["reid"].get("fastreid_root"),
        )
        body_extractor = BodyExtractor(extractor)

        multimodal_gallery = MultiModalGallery(
            face_dir=cfg["reid"].get("face_dir", "data/face"),
            body_dir=cfg["reid"].get("body_dir", "data/body"),
        )
        multimodal_gallery.build()

        gid_mgr = GlobalIDManager(
            gallery=multimodal_gallery,
            threshold=cfg["reid"]["threshold"],
            reid_interval=cfg["reid"]["reid_interval"],
        )

        if multimodal_gallery.is_empty:
            reid_warning = "ReID gallery empty (face+body)"
        else:
            logger.info(
                f"MultiModal gallery: "
                f"face={len(multimodal_gallery.face_prototypes)} "
                f"body={len(multimodal_gallery.body_prototypes)} persons"
            )

    except Exception as exc:
        reid_warning = f"MultiModal ReID unavailable: {exc}"
        logger.warning(reid_warning)

    # ── ADL buffer + PoseC3D ─────────────────────────────────────────────────
    pose_buffer = PoseSequenceBuffer(
        seq_len=cfg["adl"]["seq_len"],
        stride=cfg["adl"]["stride"],
        output_dir=cfg["adl"]["export_dir"],
        default_label=cfg["adl"].get("default_label", 0),
        max_idle_frames=cfg["adl"].get("max_idle_frames", 150),
        export_enabled=args.save_clips,
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
    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError(
            "No video source. Pass --source or put a video at data/input/ or data/sample.mp4"
        )

    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    panel_w = 220

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
    prev_track_ids: set[int] = set()
    last_query_crop = None
    last_matches: list = []
    frame_idx = -1
    metrics_interval = cfg.get("ui", {}).get("metrics_interval_frames", 5)

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

            curr_ids = {int(d.get("track_id", -1)) for d in detections if d.get("track_id", -1) >= 0}
            lost_ids = prev_track_ids - curr_ids

            # Cleanup lost tracks
            if gid_mgr is not None:
                for tid in lost_ids:
                    gid_mgr.forget_track(args.camera_id, tid)
                    if ui_logger:
                        ui_logger.log(args.camera_id, "INFO", "pipeline",
                                      f"Lost track id={tid}")
            for tid in lost_ids:
                track_status.pop(tid, None)
            prev_track_ids = curr_ids

            # ── 3+4: Body+Face ReID → GlobalID ──────────────────────────────
            for det in detections:
                tid = int(det.get("track_id", -1))
                crop, bbox_clipped = clip_bbox(frame, det["bbox"])

                # ── Face embedding (placeholder — connect ArcFace khi có) ───
                # face_feat, face_conf = arcface.extract(crop) nếu mặt nhìn thấy
                face_feat = None
                face_conf = 0.0

                # ── Assign GlobalID ──────────────────────────────────────────
                if gid_mgr is not None and tid >= 0 and crop is not None:
                    try:
                        gid, reid_score, reid_status, weights = gid_mgr.assign(
                            camera_id=args.camera_id,
                            local_track_id=tid,
                            frame=frame,
                            bbox=det["bbox"],
                            frame_idx=frame_idx,
                            face_feat=face_feat,
                            face_conf=face_conf,
                            body_extractor=body_extractor,
                        )
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID failed tid={tid}: {exc}", exc_info=True)
                        gid, reid_score, reid_status, weights = f"track_{tid}", 0.0, "reid_failed", {}

                    # Top-K matches cho panel (nếu gallery có data)
                    if multimodal_gallery is not None and not multimodal_gallery.is_empty:
                        try:
                            body_feat_panel = (
                                body_extractor.extract_from_bbox(frame, det["bbox"])
                                if body_extractor else None
                            )
                            matches_raw = multimodal_gallery.query_all(face_feat, body_feat_panel, face_conf)
                            last_matches = [(pid, sc, None) for pid, sc, _ in matches_raw[:3]]
                            last_query_crop = crop.copy() if crop is not None else None
                        except Exception:
                            pass
                else:
                    gid = f"track_{tid}" if tid >= 0 else "unknown"
                    reid_score, reid_status, weights = 0.0, "reid_unavailable", {}
                    if crop is not None:
                        last_query_crop = crop.copy()

                # ── 5: ADL buffer ────────────────────────────────────────────
                adl_status = {"status": "waiting", "current_len": 0,
                              "seq_len": cfg["adl"]["seq_len"], "pkl_path": None}
                if tid >= 0:
                    adl_status = pose_buffer.update(
                        args.camera_id, tid, gid, frame_idx,
                        det.get("keypoints"), det.get("keypoint_scores"), (h, w),
                    )

                    if adl_status and adl_status.get("status") == "exported":
                        event_bus.emit("pose_clip_exported", {
                            "camera_id": args.camera_id, "frame_idx": frame_idx,
                            "local_track_id": tid, "global_id": gid,
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
                                        "camera_id": args.camera_id, "frame_idx": frame_idx,
                                        "local_track_id": tid, "global_id": gid,
                                        "label": result["label"],
                                        "score": float(result.get("score", 0.0)),
                                    })
                                    if ui_logger:
                                        ui_logger.log(
                                            args.camera_id, "INFO", "adl",
                                            f"ADL: gid={gid} action={result['label']} "
                                            f"score={result.get('score', 0.0):.2f}",
                                        )
                                elif isinstance(result, dict):
                                    adl_status = {**result, "pkl_path": adl_status.get("pkl_path")}
                            except Exception as exc:
                                logger.warning(f"PoseC3D failed: {exc}", exc_info=True)
                                adl_status = {"status": "failed", "label": None,
                                              "score": 0.0, "pkl_path": adl_status.get("pkl_path")}

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
            if ui_logger and frame_idx % metrics_interval == 0:
                ui_logger.metric(args.camera_id, {
                    "camera_id": args.camera_id,
                    "module": "pipeline",
                    "frame_idx": frame_idx,
                    "fps": round(current_fps, 1),
                    "device": device,
                    "persons": len(detections),
                    "tracked": len(curr_ids),
                    "reid_assigned": sum(
                        1 for v in gid_mgr._cache.values()
                        if "gid_" in v.get("global_id", "")
                    ) if gid_mgr else 0,
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
            face_count = len(multimodal_gallery.face_prototypes) if multimodal_gallery else 0
            body_count = len(multimodal_gallery.body_prototypes) if multimodal_gallery else 0
            info: dict = {
                "Module": "Full CPose (Face+Body ReID)",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Persons": len(detections),
                "Gallery F/B": f"{face_count}/{body_count}",
                "Device": device,
                "FPS": f"{current_fps:.1f}",
            }
            if reid_warning:
                info["Warning"] = reid_warning[:48]

            draw_info_panel(frame, info)

            first_status = next(iter(track_status.values()), {
                "status": "waiting", "current_len": 0, "seq_len": cfg["adl"]["seq_len"],
            })
            draw_adl_status(frame, first_status, pos=(10, 170))
            display = draw_reid_panel(frame, last_query_crop, last_matches, panel_w=panel_w)

            # ── Output ───────────────────────────────────────────────────────
            if writer is not None:
                writer.write(display)
            if show:
                key = safe_imshow(WINDOW_NAME, display)
                if key in (27, ord("q"), ord("Q")):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Pipeline finished. Processed {frame_idx + 1} frames.")


if __name__ == "__main__":
    main()
