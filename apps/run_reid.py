"""
apps/run_reid.py — Debug ReID với visual gallery panel

Có 2 chế độ:
  1) Body ReID (FastReID): cần .github/fast-reid và model weights
  2) Face gallery browser: load .npy embeddings từ data/face/ (không cần FastReID)

Chạy:
    # Chế độ 1: Body ReID trên video
    python apps/run_reid.py --source data/input/cam1.mp4

    # Chế độ 2: Query ảnh đơn lẻ vào gallery
    python apps/run_reid.py --query data/gallery/someone/img.jpg

    # Chế độ face gallery browser (auto nếu FastReID không khả dụng)
    python apps/run_reid.py --source data/input/cam1.mp4 --face-gallery

Phím:
    Q / ESC : thoát
    S       : screenshot
    SPACE   : pause
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_pipeline_cfg
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel, draw_reid_panel

logger = get_logger(__name__)

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def find_default_source(root: Path) -> str | None:
    for ext in VIDEO_EXTS:
        files = sorted((root / "data" / "input").glob(ext))
        if files:
            return str(files[0])
    return None


# ──────────────────────────────────────────────────────────────
# Face Gallery (load .npy embeddings — không cần FastReID)
# ──────────────────────────────────────────────────────────────

class FaceGallery:
    """Load pre-extracted face embeddings từ data/face/<person_id>/ structure."""

    def __init__(self, face_dir: Path):
        self.face_dir  = face_dir
        self.prototypes: dict[str, np.ndarray] = {}
        self.images:     dict[str, list[np.ndarray]] = {}   # thumbnail BGR

    def load(self):
        """Load tất cả embeddings và ảnh thumbnail."""
        self.prototypes = {}
        self.images = {}
        if not self.face_dir.exists():
            logger.warning(f"Face gallery dir không tồn tại: {self.face_dir}")
            return

        for person_dir in sorted(self.face_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            # Load embeddings (.npy)
            emb_files = sorted(person_dir.glob("*.npy"))
            embs = []
            for ef in emb_files:
                arr = np.load(str(ef))
                if arr.ndim == 1:
                    embs.append(arr.astype(np.float32))
                elif arr.ndim == 2:
                    embs.extend(arr.astype(np.float32))

            if not embs:
                logger.warning(f"Không tìm thấy .npy trong: {person_dir}")
                continue

            # Prototype = L2-normalized mean
            stack = np.stack(embs, axis=0)
            proto = stack.mean(axis=0)
            norm  = np.linalg.norm(proto)
            if norm > 1e-6:
                proto = proto / norm
            self.prototypes[person_dir.name] = proto.astype(np.float32)

            # Ảnh thumbnail (nếu có)
            thumbs = []
            for ext in IMAGE_EXTS:
                for p in person_dir.glob(ext):
                    img = cv2.imread(str(p))
                    if img is not None:
                        thumbs.append(cv2.resize(img, (60, 80)))
            self.images[person_dir.name] = thumbs

            logger.info(
                f"  Loaded person '{person_dir.name}': "
                f"{len(embs)} embeddings, {len(thumbs)} images"
            )

        logger.info(
            f"Face gallery: {len(self.prototypes)} persons loaded from {self.face_dir}"
        )

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def query(self, feat: np.ndarray, topk: int = 3) -> list[tuple[str, float]]:
        """Trả list[(person_id, score)] sorted descending."""
        scores = []
        for pid, proto in self.prototypes.items():
            scores.append((pid, self._cosine(feat, proto)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


# ──────────────────────────────────────────────────────────────
# Body ReID panel on video
# ──────────────────────────────────────────────────────────────

def run_body_reid_video(args, cfg):
    """Chạy FastReID trên video, hiển thị gallery panel."""
    from src.reid.fast_reid import FastReIDExtractor
    from src.reid.gallery import ReIDGallery
    from src.detectors.yolo_pose import YoloPoseTracker
    from src.trackers.bytetrack import ByteTrackWrapper

    logger.info("Loading FastReID extractor...")
    extractor = FastReIDExtractor(
        fastreid_root=cfg["reid"]["fastreid_root"],
        config_path=cfg["reid"]["config"],
        weights_path=cfg["reid"]["weights"],
        device=cfg["system"]["device"],
    )
    gallery = ReIDGallery(extractor, cfg["reid"]["gallery_dir"])
    gallery.build()
    logger.info(f"Gallery: {len(gallery.prototypes)} persons")

    logger.info("Loading YOLO...")
    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"],
    )
    tracker = ByteTrackWrapper(detector)

    source = args.source or find_default_source(ROOT)
    if source is None:
        logger.error("Không tìm thấy video. Dùng --source")
        sys.exit(1)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        sys.exit(1)

    src_name    = Path(source).stem if isinstance(source, str) else "webcam"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_ctr     = FPSCounter()
    frame_idx   = 0

    # Giữ matches của track gần nhất để hiện panel
    last_query_crop: np.ndarray | None = None
    last_matches: list = []

    logger.info("=" * 50)
    logger.info("  Q/ESC: thoát | SPACE: pause | S: screenshot")
    logger.info("=" * 50)

    paused = False
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
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            try:
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracker failed: {exc}", exc_info=True)
                continue

            for det in detections:
                tid = det.get("track_id", -1)
                if tid < 0:
                    continue

                x1, y1, x2, y2 = map(int, det["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                try:
                    feat = extractor.extract(crop)
                    matches_raw = []
                    for pid, proto in gallery.prototypes.items():
                        sim = float(np.dot(feat, proto) /
                                    (np.linalg.norm(feat) * np.linalg.norm(proto) + 1e-12))
                        matches_raw.append((pid, sim))
                    matches_raw.sort(key=lambda x: x[1], reverse=True)

                    top_id, top_score = matches_raw[0] if matches_raw else ("?", 0.0)
                    label = f"{top_id}|{top_score:.2f}"
                    draw_detection(frame, det, label=label)

                    # Cập nhật panel với track cuối
                    if tid == detections[-1].get("track_id"):
                        last_query_crop = crop.copy()
                        last_matches = [
                            (pid, score, None) for pid, score in matches_raw[:3]
                        ]
                except Exception as exc:
                    logger.warning(f"ReID extract failed: {exc}", exc_info=True)
                    draw_detection(frame, det, label=f"ID:{tid}")

            fps_val = fps_ctr.tick()
            draw_info_panel(frame, {
                "Run ReID":  "(Body)",
                "Frame":     f"{frame_idx}/{total_frames}",
                "FPS":       f"{fps_val:.1f}",
                "Persons":   len(detections),
                "Gallery":   f"{len(gallery.prototypes)} persons",
            })

            # Gắn ReID panel bên phải
            display = draw_reid_panel(frame, last_query_crop, last_matches, panel_w=200)

            cv2.imshow("CPose — Run ReID (Body)  [Q/ESC: thoát]", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                paused = True
            elif key in (ord('s'), ord('S')):
                ss_dir  = ensure_dir(ROOT / "data" / "output" / "screenshots")
                ss_path = str(ss_dir / f"reid_{src_name}_f{frame_idx:06d}.jpg")
                cv2.imwrite(ss_path, display)
                logger.info(f"Screenshot: {ss_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# Face gallery browser (không cần FastReID)
# ──────────────────────────────────────────────────────────────

def run_face_gallery_browser(args, cfg):
    """
    Hiển thị face gallery. Chạy YOLO để lấy person crops,
    so sánh với face embeddings bằng cosine similarity.

    Lưu ý: chỉ hoạt động đúng nếu embeddings trong data/face/ được
    extract bằng CÙNG model. Dùng để debug gallery, không phải production.
    """
    face_dir = ROOT / "data" / "face"
    fg = FaceGallery(face_dir)
    fg.load()

    if not fg.prototypes:
        logger.error(f"Không tìm thấy embedding nào trong {face_dir}")
        logger.error("Cần có file .npy trong data/face/<person_name>/emb_XX.npy")
        sys.exit(1)

    # Xây dựng gallery display grid
    logger.info("Building gallery display...")
    gallery_display = _build_gallery_grid(fg, cell_w=80, cell_h=100)

    from src.detectors.yolo_pose import YoloPoseTracker
    from src.trackers.bytetrack import ByteTrackWrapper

    detector = YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracker"]["tracker_yaml"],
        device=cfg["system"]["device"],
    )
    tracker = ByteTrackWrapper(detector)

    source = args.source or find_default_source(ROOT)
    if source is None:
        # Không có video: chỉ hiển thị gallery browser
        logger.info("Không có video source. Chỉ hiển thị gallery.")
        _show_gallery_only(gallery_display, fg)
        return

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        sys.exit(1)

    src_name    = Path(source).stem if isinstance(source, str) else "webcam"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_ctr     = FPSCounter()
    frame_idx   = 0
    last_matches: list = []
    last_crop: np.ndarray | None = None
    paused = False

    logger.info("=" * 55)
    logger.info(f"  Face gallery: {list(fg.prototypes.keys())}")
    logger.info("  NOTE: Kết quả chỉ đúng nếu embeddings dùng cùng model")
    logger.info("  Q/ESC: thoát | SPACE: pause | S: screenshot | G: show gallery")
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
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            try:
                detections, _ = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] tracker failed: {exc}", exc_info=True)
                continue

            for det in detections:
                tid = det.get("track_id", -1)
                if tid < 0:
                    continue
                x1, y1, x2, y2 = map(int, det["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Chú ý: không có face extractor → không thể extract embedding mới
                # Hiển thị person crop với label "No face model"
                label = f"ID:{tid} [no face model]"
                draw_detection(frame, det, label=label)

                if tid == detections[-1].get("track_id"):
                    last_crop = crop.copy()
                    last_matches = [(pid, 0.0, None) for pid in list(fg.prototypes.keys())[:3]]

            fps_val = fps_ctr.tick()
            draw_info_panel(frame, {
                "Run ReID":    "(Face Gallery Browser)",
                "Frame":       f"{frame_idx}/{total_frames}",
                "FPS":         f"{fps_val:.1f}",
                "Persons":     len(detections),
                "Face persons": ", ".join(fg.prototypes.keys()),
                "Note":        "Install face extractor",
            })

            display = draw_reid_panel(frame, last_crop, last_matches, panel_w=220)
            cv2.imshow("CPose — Face Gallery Browser  [Q: thoát | G: gallery]", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                paused = True
            elif key in (ord('g'), ord('G')):
                cv2.imshow("CPose — Face Gallery", gallery_display)
            elif key in (ord('s'), ord('S')):
                ss_dir  = ensure_dir(ROOT / "data" / "output" / "screenshots")
                ss_path = str(ss_dir / f"face_gallery_{src_name}_f{frame_idx:06d}.jpg")
                cv2.imwrite(ss_path, display)
                logger.info(f"Screenshot: {ss_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def _build_gallery_grid(fg: FaceGallery, cell_w=80, cell_h=100) -> np.ndarray:
    """Tạo ảnh grid hiển thị tất cả persons trong gallery."""
    persons = sorted(fg.prototypes.keys())
    n = len(persons)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    grid_w = cols * (cell_w + 4) + 4
    grid_h = rows * (cell_h + 20 + 4) + 4
    grid   = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid[:] = (40, 40, 40)

    for idx, person_id in enumerate(persons):
        row, col = divmod(idx, cols)
        x0 = col * (cell_w + 4) + 4
        y0 = row * (cell_h + 20 + 4) + 4

        thumbs = fg.images.get(person_id, [])
        if thumbs:
            thumb = cv2.resize(thumbs[0], (cell_w, cell_h))
        else:
            thumb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            thumb[:] = (60, 60, 60)
            cv2.putText(thumb, "No img", (4, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        grid[y0:y0 + cell_h, x0:x0 + cell_w] = thumb
        # Name label
        cv2.putText(grid, person_id[:10], (x0, y0 + cell_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    return grid


def _show_gallery_only(grid: np.ndarray, fg: FaceGallery):
    cv2.imshow("CPose — Face Gallery [Q/ESC: thoát]", grid)
    logger.info(f"Gallery persons: {list(fg.prototypes.keys())}")
    logger.info("Nhấn Q hoặc ESC để thoát.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# Query mode (ảnh đơn lẻ)
# ──────────────────────────────────────────────────────────────

def run_query_mode(args, cfg):
    """Query 1 ảnh vào body gallery, hiển thị kết quả."""
    from src.reid.fast_reid import FastReIDExtractor
    from src.reid.gallery import ReIDGallery

    extractor = FastReIDExtractor(
        fastreid_root=cfg["reid"]["fastreid_root"],
        config_path=cfg["reid"]["config"],
        weights_path=cfg["reid"]["weights"],
        device=cfg["system"]["device"],
    )
    gallery = ReIDGallery(extractor, cfg["reid"]["gallery_dir"])
    gallery.build()

    query_img = cv2.imread(args.query)
    if query_img is None:
        logger.error(f"Không đọc được ảnh: {args.query}")
        sys.exit(1)

    feat = extractor.extract(query_img)
    scores = []
    for pid, proto in gallery.prototypes.items():
        sim = float(np.dot(feat, proto) / (np.linalg.norm(feat) * np.linalg.norm(proto) + 1e-12))
        scores.append((pid, sim))
    scores.sort(key=lambda x: x[1], reverse=True)

    top_matches = [(pid, sc, None) for pid, sc in scores[:args.topk]]
    display = draw_reid_panel(
        cv2.resize(query_img, (400, 500)),
        query_img, top_matches, panel_w=220,
    )
    logger.info(f"Query: {args.query}")
    for pid, sc, _ in top_matches:
        logger.info(f"  {sc:.4f}  {pid}")

    cv2.imshow("CPose — ReID Query Result  [Q/ESC: thoát]", display)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CPose — Run ReID: visual gallery debug")
    parser.add_argument("--source", type=str, default=None, help="Video path hoặc webcam index")
    parser.add_argument("--query",  type=str, default=None, help="Query image path (chế độ single-query)")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    parser.add_argument("--topk",       type=int,  default=3, help="Số kết quả top-k")
    parser.add_argument("--face-gallery", action="store_true",
                        help="Dùng face gallery (.npy) thay vì FastReID body gallery")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_pipeline_cfg(Path(args.config), ROOT)

    # Single query mode
    if args.query:
        try:
            run_query_mode(args, cfg)
        except Exception as exc:
            logger.error(f"Query mode failed: {exc}")
        return

    # Face gallery mode
    if args.face_gallery:
        run_face_gallery_browser(args, cfg)
        return

    # Body ReID video mode — fallback to face gallery browser nếu FastReID missing
    try:
        run_body_reid_video(args, cfg)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        logger.warning(f"FastReID không khả dụng: {exc}")
        logger.warning("Chuyển sang Face Gallery Browser mode...")
        run_face_gallery_browser(args, cfg)


if __name__ == "__main__":
    main()