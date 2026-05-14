"""
apps/run_pipeline.py — Full CPose pipeline với real-time display

Chạy:
    python apps/run_pipeline.py                              # auto-find video
    python apps/run_pipeline.py --source data/input/cam1.mp4
    python apps/run_pipeline.py --source 0 --camera-id cam01
    python apps/run_pipeline.py --source data/input/cam1.mp4 --save

Phím:
    Q / ESC : thoát
    S       : screenshot
    SPACE   : pause / resume
    I       : toggle info panel
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.pose_buffer import PoseSequenceBuffer
from src.core.event import EventBus
from src.core.global_id import GlobalIDManager
from src.detectors.yolo_pose import YoloPoseTracker
from src.trackers.bytetrack import ByteTrackWrapper
from src.utils.config import load_pipeline_cfg
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel, track_color

logger = get_logger("pipeline")

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")


def find_default_source(root: Path) -> str | None:
    input_dir = root / "data" / "input"
    if not input_dir.exists():
        return None
    for ext in VIDEO_EXTS:
        files = sorted(input_dir.glob(ext))
        if files:
            return str(files[0])
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="CPose full inference pipeline")
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video path hoặc webcam index. Mặc định: auto-find trong data/input/",
    )
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    parser.add_argument("--no-show", action="store_true", help="Tắt cửa sổ display")
    parser.add_argument("--save", action="store_true", help="Lưu video output")
    parser.add_argument(
        "--no-reid", action="store_true",
        help="Bỏ qua ReID (dùng track_id làm global_id). Hữu ích khi chưa cài FastReID.",
    )
    return parser.parse_args()


def try_build_reid(cfg):
    """Cố build ReID stack, trả về (gallery, global_id_manager) hoặc (None, None)."""
    try:
        from src.reid.fast_reid import FastReIDExtractor
        from src.reid.gallery import ReIDGallery

        reid_extractor = FastReIDExtractor(
            fastreid_root=cfg["reid"]["fastreid_root"],
            config_path=cfg["reid"]["config"],
            weights_path=cfg["reid"]["weights"],
            device=cfg["system"]["device"],
        )
        gallery = ReIDGallery(reid_extractor, cfg["reid"]["gallery_dir"])
        gallery.build()
        gid_mgr = GlobalIDManager(
            gallery,
            threshold=cfg["reid"]["threshold"],
            reid_interval=cfg["reid"].get("reid_interval", 10),
        )
        logger.info("FastReID loaded ✓")
        return gid_mgr, True
    except FileNotFoundError as exc:
        logger.warning(f"FastReID không khả dụng: {exc}")
        logger.warning("Chạy ở chế độ tracking-only (không ReID). Dùng --no-reid để tắt cảnh báo này.")
        return None, False
    except Exception as exc:
        logger.warning(f"FastReID load failed: {exc}")
        return None, False


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)

    # ── Detector + Tracker ──
    logger.info("Building detector...")
    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"],
    )
    tracker = ByteTrackWrapper(detector)

    # ── ReID (optional) ──
    gid_mgr   = None
    reid_ok   = False
    if not args.no_reid:
        gid_mgr, reid_ok = try_build_reid(cfg)

    # ── PoseBuffer + EventBus ──
    pose_buffer = PoseSequenceBuffer(
        seq_len=cfg["adl"]["seq_len"],
        stride=cfg["adl"]["stride"],
        output_dir=cfg["adl"]["export_dir"],
        default_label=cfg["adl"]["default_label"],
        max_idle_frames=cfg["adl"].get("max_idle_frames", 150),
    )
    event_bus = EventBus(cfg["system"]["event_log"])

    # ── PoseC3D (optional) ──
    posec3d_runner = None
    if cfg["adl"].get("auto_infer", False):
        try:
            from src.action.posec3d import PoseC3DRunner
            posec3d_runner = PoseC3DRunner(
                mmaction_root=cfg["adl"]["mmaction_root"],
                base_config=cfg["adl"]["base_config"],
                checkpoint=cfg["adl"]["weights"],
                work_dir=cfg["adl"]["work_dir"],
            )
            logger.info("PoseC3D runner loaded ✓")
        except Exception as exc:
            logger.warning(f"PoseC3D không khả dụng: {exc}")

    # ── Source ──
    source = args.source
    if source is None:
        source = find_default_source(ROOT)
        if source is None:
            logger.error(
                "Không tìm thấy video trong data/input/\n"
                "Dùng: python apps/run_pipeline.py --source <đường_dẫn>"
            )
            sys.exit(1)
        logger.info(f"Auto-detected source: {source}")
    else:
        if source.isdigit():
            source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        sys.exit(1)

    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_name     = Path(source).stem if isinstance(source, str) else "webcam"

    logger.info(
        f"Source: {source} | {src_w}×{src_h} | {src_fps:.1f}fps | "
        f"Camera: {args.camera_id} | ReID: {'ON' if reid_ok else 'OFF (tracking-only)'}"
    )

    # ── VideoWriter ──
    writer = None
    if args.save:
        out_dir  = ensure_dir(ROOT / cfg["system"]["vis_dir"])
        out_path = str(out_dir / f"{args.camera_id}_{src_name}_pipeline.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (src_w, src_h))
        logger.info(f"Saving to: {out_path}")

    show      = not args.no_show
    paused    = False
    show_info = True
    fps_ctr   = FPSCounter(alpha=0.15)
    frame_idx = -1

    # Track active IDs để cleanup GlobalIDManager
    prev_track_ids: set[int] = set()

    logger.info("=" * 55)
    logger.info("  Q/ESC  : thoát   |  SPACE : pause   |  I : info panel")
    logger.info("  S      : screenshot")
    logger.info(f"  Mode   : {'Full pipeline (YOLO+ByteTrack+ReID)' if reid_ok else 'Tracking-only (YOLO+ByteTrack)'}")
    logger.info("=" * 55)

    try:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if key == ord(' '):
                    paused = False
                continue

            ok, frame = cap.read()
            if not ok:
                logger.info("Hết video.")
                while True:
                    key = cv2.waitKey(500) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    if key in (ord('r'), ord('R')):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = -1
                        break
                else:
                    break
                continue

            frame_idx += 1
            h, w = frame.shape[:2]

            # ── Tracking ──
            try:
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracker failed: {exc}", exc_info=True)
                continue

            # ── Cleanup lost tracks trong GlobalIDManager ──
            if gid_mgr is not None:
                curr_ids = {det["track_id"] for det in detections if det.get("track_id", -1) >= 0}
                for lost_tid in prev_track_ids - curr_ids:
                    gid_mgr.forget_track(args.camera_id, lost_tid)
                prev_track_ids = curr_ids

            # ── Per-detection ──
            for det in detections:
                tid = det.get("track_id", -1)
                if tid < 0:
                    continue

                # Crop
                x1, y1, x2, y2 = map(int, det["bbox"])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # ReID hoặc fallback
                if gid_mgr is not None:
                    try:
                        global_id, reid_score = gid_mgr.assign(
                            camera_id=args.camera_id,
                            local_track_id=tid,
                            crop_bgr=crop,
                            frame_idx=frame_idx,
                        )
                    except Exception as exc:
                        logger.warning(f"ReID failed track {tid}: {exc}", exc_info=True)
                        global_id, reid_score = f"t{tid}", 0.0
                else:
                    # tracking-only: dùng track_id làm global_id
                    global_id, reid_score = f"t{tid}", 1.0

                # PoseBuffer
                pkl_path = pose_buffer.update(
                    camera_id=args.camera_id,
                    local_track_id=tid,
                    global_id=global_id,
                    frame_idx=frame_idx,
                    keypoints_xy=det.get("keypoints"),
                    keypoint_scores=det.get("keypoint_scores"),
                    img_shape=(h, w),
                )

                # Vẽ
                if reid_ok:
                    label = f"{global_id} | {reid_score:.2f}"
                else:
                    label = f"ID:{global_id}"
                draw_detection(frame, det, label=label)

                # Events
                event_bus.emit("track_update", {
                    "camera_id":    args.camera_id,
                    "frame_idx":    frame_idx,
                    "local_track_id": int(tid),
                    "global_id":    global_id,
                    "reid_score":   float(reid_score),
                    "bbox":         [x1, y1, x2, y2],
                })

                if pkl_path is not None:
                    event_bus.emit("pose_clip_exported", {
                        "camera_id":    args.camera_id,
                        "frame_idx":    frame_idx,
                        "local_track_id": int(tid),
                        "global_id":    global_id,
                        "pkl_path":     str(pkl_path),
                    })
                    if posec3d_runner is not None:
                        try:
                            posec3d_runner.run_test(str(pkl_path))
                        except Exception as exc:
                            logger.warning(f"PoseC3D failed: {exc}", exc_info=True)

            # ── Info panel ──
            fps_val = fps_ctr.tick()
            if show_info:
                progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)
                mode_str = "YOLO+ByteTrack+ReID" if reid_ok else "YOLO+ByteTrack"
                draw_info_panel(frame, {
                    "Run Pipeline": "",
                    "Camera":  args.camera_id,
                    "Frame":   progress,
                    "FPS":     f"{fps_val:.1f}",
                    "Persons": len(detections),
                    "Mode":    mode_str[:22],
                    "Source":  src_name[:20],
                })

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow(
                    "CPose — Pipeline  [Q: thoát | SPACE: pause | I: info | S: screenshot]",
                    frame,
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("Người dùng thoát.")
                    break
                elif key == ord(' '):
                    paused = True
                    logger.info("PAUSED")
                elif key in (ord('i'), ord('I')):
                    show_info = not show_info
                elif key in (ord('s'), ord('S')):
                    ss_dir  = ensure_dir(ROOT / "data" / "output" / "screenshots")
                    ss_path = str(ss_dir / f"pipeline_{args.camera_id}_f{frame_idx:06d}.jpg")
                    cv2.imwrite(ss_path, frame)
                    logger.info(f"Screenshot: {ss_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Pipeline stopped. Frames: {frame_idx} | FPS avg: {fps_ctr.fps:.1f}")
        if args.save and writer is not None:
            logger.info(f"Video đã lưu: {out_path}")


if __name__ == "__main__":
    main()