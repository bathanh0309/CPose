"""
apps/run_pose.py — Debug YOLO Pose với real-time display

Chạy:
    python apps/run_pose.py                          # auto-find video trong data/input/
    python apps/run_pose.py --source 0               # webcam
    python apps/run_pose.py --source data/input/cam1.mp4
    python apps/run_pose.py --source data/input/cam1.mp4 --save

Phím:
    Q / ESC : thoát
    S       : chụp screenshot
    SPACE   : pause / resume
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_pipeline_cfg
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel

logger = get_logger(__name__)

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")


def find_default_source(root: Path) -> str | None:
    """Tự tìm video đầu tiên trong data/input/."""
    input_dir = root / "data" / "input"
    if not input_dir.exists():
        return None
    for ext in VIDEO_EXTS:
        files = sorted(input_dir.glob(ext))
        if files:
            return str(files[0])
    return None


def list_videos(root: Path) -> list[str]:
    """Liệt kê tất cả video trong data/input/."""
    input_dir = root / "data" / "input"
    videos = []
    if input_dir.exists():
        for ext in VIDEO_EXTS:
            videos.extend(sorted(input_dir.glob(ext)))
    return [str(p) for p in videos]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPose — Run Pose: YOLO pose detection + tracking với real-time display"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video/image path hoặc webcam index. Mặc định: auto-find trong data/input/",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    parser.add_argument("--conf", type=float, default=None, help="Override confidence (default từ config)")
    parser.add_argument("--no-show", action="store_true", help="Tắt cửa sổ hiển thị")
    parser.add_argument("--save", action="store_true", help="Lưu video output vào data/output/vis/")
    parser.add_argument("--list", action="store_true", help="Liệt kê video trong data/input/ rồi thoát")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── List mode ──
    if args.list:
        videos = list_videos(ROOT)
        if not videos:
            logger.info("Không tìm thấy video nào trong data/input/")
        else:
            logger.info(f"Tìm thấy {len(videos)} video trong data/input/:")
            for v in videos:
                logger.info(f"  {v}")
        return

    # ── Load config ──
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    weights = cfg["pose"]["weights"]
    conf    = args.conf or cfg["pose"]["conf"]
    iou     = cfg["pose"]["iou"]
    device  = cfg["system"]["device"]

    # ── Load YOLO ──
    from ultralytics import YOLO
    logger.info(f"Loading YOLO model: {weights}")
    model = YOLO(weights)

    # ── Chọn source ──
    source = args.source
    if source is None:
        source = find_default_source(ROOT)
        if source is None:
            logger.error(
                "Không tìm thấy video trong data/input/\n"
                "  Dùng: python apps/run_pose.py --source <đường_dẫn_video>\n"
                "  Hoặc copy video vào thư mục: data/input/"
            )
            sys.exit(1)
        logger.info(f"Auto-detected source: {source}")
    else:
        if source.isdigit():
            source = int(source)

    # ── Mở VideoCapture ──
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Không thể mở source: {source}")
        sys.exit(1)

    src_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_name    = Path(source).stem if isinstance(source, str) else "webcam"

    logger.info(
        f"Source: {source}\n"
        f"  Resolution : {src_w}×{src_h}\n"
        f"  FPS        : {src_fps:.1f}\n"
        f"  Frames     : {total_frames if total_frames > 0 else 'unknown'}"
    )

    # ── VideoWriter ──
    writer = None
    if args.save:
        out_dir  = ensure_dir(ROOT / cfg["system"]["vis_dir"])
        out_path = str(out_dir / f"{src_name}_pose.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (src_w, src_h))
        logger.info(f"Saving output to: {out_path}")

    show    = not args.no_show
    paused  = False
    fps_ctr = FPSCounter(alpha=0.15)
    frame_idx = 0

    logger.info("=" * 50)
    logger.info("  Q / ESC  : thoát")
    logger.info("  S        : chụp screenshot")
    logger.info("  SPACE    : pause / resume")
    logger.info("=" * 50)

    try:
        while True:
            # ── Pause ──
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if key == ord(' '):
                    paused = False
                continue

            ok, frame = cap.read()
            if not ok:
                logger.info("Hết video. Chạy lại? (nhấn R) hoặc thoát (nhấn Q)")
                # Chờ user nhấn phím
                while True:
                    key = cv2.waitKey(500) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    if key in (ord('r'), ord('R')):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        break
                else:
                    break
                continue

            frame_idx += 1

            # ── YOLO tracking inference ──
            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=cfg["tracker"]["tracker_yaml"],
                    conf=conf,
                    iou=iou,
                    classes=[0],   # person only
                    device=device,
                    verbose=False,
                )
                result = results[0]
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] YOLO failed: {exc}", exc_info=True)
                continue

            # ── Parse detections ──
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy    = result.boxes.xyxy.cpu().numpy()
                confs   = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                track_ids = (
                    result.boxes.id.cpu().numpy().astype(int)
                    if result.boxes.id is not None else None
                )
                kp_xy   = None
                kp_conf = None
                if result.keypoints is not None:
                    kp_xy = result.keypoints.xy.cpu().numpy()
                    if result.keypoints.conf is not None:
                        kp_conf = result.keypoints.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    detections.append({
                        "bbox":            xyxy[i].tolist(),
                        "score":           float(confs[i]),
                        "class_id":        int(cls_ids[i]),
                        "track_id":        int(track_ids[i]) if track_ids is not None else -1,
                        "keypoints":       kp_xy[i].tolist() if kp_xy is not None else None,
                        "keypoint_scores": kp_conf[i].tolist() if kp_conf is not None else None,
                    })

            # ── Vẽ ──
            for det in detections:
                tid   = det.get("track_id", -1)
                label = f"ID:{tid}" if tid >= 0 else f"conf:{det['score']:.2f}"
                draw_detection(frame, det, label=label)

            # ── Info panel ──
            fps_val = fps_ctr.tick()
            progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)
            draw_info_panel(frame, {
                "Run Pose": "",
                "Frame":   progress,
                "FPS":     f"{fps_val:.1f}",
                "Persons": len(detections),
                "Conf":    f"{conf:.2f}",
                "Source":  src_name[:20],
            })

            # ── Ghi file ──
            if writer is not None:
                writer.write(frame)

            # ── Hiển thị ──
            if show:
                cv2.imshow("CPose — Run Pose  [Q/ESC: thoát | SPACE: pause | S: screenshot]", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("Người dùng thoát.")
                    break
                elif key == ord(' '):
                    paused = True
                    logger.info("PAUSED — nhấn SPACE để tiếp tục")
                elif key in (ord('s'), ord('S')):
                    ss_dir  = ensure_dir(ROOT / "data" / "output" / "screenshots")
                    ss_path = str(ss_dir / f"{src_name}_f{frame_idx:06d}.jpg")
                    cv2.imwrite(ss_path, frame)
                    logger.info(f"Screenshot: {ss_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(
            f"Hoàn thành. Đã xử lý {frame_idx} frames | "
            f"FPS trung bình: {fps_ctr.fps:.1f}"
        )
        if args.save and writer is not None:
            logger.info(f"Video đã lưu tại: {out_path}")


if __name__ == "__main__":
    main()