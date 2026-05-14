"""
apps/run_adl.py — Debug ADL/PoseC3D với skeleton animation

Có 2 chế độ:
  1) Visualize pkl: load file .pkl (MMAction2 format) và animate skeleton
  2) Run inference: gọi PoseC3D subprocess (cần MMAction2)

Chạy:
    # Xem skeleton animation từ pkl (không cần MMAction2)
    python apps/run_adl.py --viz data/output/clips_pkl/sample.pkl

    # Auto-find pkl mới nhất trong output dir
    python apps/run_adl.py --viz-latest

    # Run PoseC3D inference (cần MMAction2 setup)
    python apps/run_adl.py --input data/output/clips_pkl/sample.pkl

Phím (khi visualize):
    Q / ESC  : thoát
    SPACE    : pause / resume
    → / ←   : next / prev frame
    R        : restart animation
    S        : screenshot
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_pipeline_cfg
from src.utils.io import ensure_dir, load_pickle
from src.utils.logger import get_logger
from src.utils.vis import COCO_EDGES, COCO_JOINT_NAMES, EDGE_COLORS, draw_info_panel

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Skeleton canvas renderer
# ──────────────────────────────────────────────────────────────

CANVAS_W = 640
CANVAS_H = 480
BG_COLOR  = (20, 20, 30)      # dark background

JOINT_COLOR  = (255, 255, 255)
JOINT_RADIUS = 5


def normalize_keypoints(kpts: np.ndarray, canvas_w: int, canvas_h: int, margin: float = 0.1) -> np.ndarray:
    """Scale keypoints (T, V, 2) vào canvas với margin."""
    kpts = kpts.copy()
    valid = kpts[(kpts[:, :, 0] > 1) & (kpts[:, :, 1] > 1)]
    if len(valid) < 2:
        return kpts

    x_min, y_min = valid[:, 0].min(), valid[:, 1].min()
    x_max, y_max = valid[:, 0].max(), valid[:, 1].max()

    rx = x_max - x_min
    ry = y_max - y_min
    if rx < 1:
        rx = 1.0
    if ry < 1:
        ry = 1.0

    scale = min(
        canvas_w * (1 - 2 * margin) / rx,
        canvas_h * (1 - 2 * margin) / ry,
    )

    kpts[:, :, 0] = (kpts[:, :, 0] - x_min) * scale + canvas_w * margin
    kpts[:, :, 1] = (kpts[:, :, 1] - y_min) * scale + canvas_h * margin
    return kpts


def draw_skeleton_frame(
    canvas: np.ndarray,
    kpts_t: np.ndarray,         # [V, 2]
    scores_t: np.ndarray | None,  # [V] or None
    color_hint: tuple = (0, 220, 180),
    thickness: int = 2,
):
    """Vẽ 1 frame skeleton lên canvas."""
    V = kpts_t.shape[0]
    scores = scores_t if scores_t is not None else np.ones(V)

    # Edges
    for idx, (a, b) in enumerate(COCO_EDGES):
        if a >= V or b >= V:
            continue
        sa = float(scores[a])
        sb = float(scores[b])
        if sa < 0.2 or sb < 0.2:
            continue
        if kpts_t[a, 0] < 1 or kpts_t[b, 0] < 1:
            continue
        xa, ya = int(kpts_t[a, 0]), int(kpts_t[a, 1])
        xb, yb = int(kpts_t[b, 0]), int(kpts_t[b, 1])
        ec = EDGE_COLORS[idx] if idx < len(EDGE_COLORS) else color_hint
        cv2.line(canvas, (xa, ya), (xb, yb), ec, thickness, cv2.LINE_AA)

    # Joints
    for v in range(V):
        if float(scores[v]) < 0.2:
            continue
        x, y = int(kpts_t[v, 0]), int(kpts_t[v, 1])
        if x < 1 or y < 1:
            continue
        cv2.circle(canvas, (x, y), JOINT_RADIUS, JOINT_COLOR, -1)
        cv2.circle(canvas, (x, y), JOINT_RADIUS - 1, color_hint, -1)


def draw_trajectory(
    canvas: np.ndarray,
    kpts_all: np.ndarray,       # [T, V, 2]
    current_t: int,
    trail_len: int = 10,
    joint_idx: int = 0,         # nose hoặc centroid
):
    """Vẽ trajectory của 1 joint qua các frame."""
    t_start = max(0, current_t - trail_len)
    for t in range(t_start, current_t):
        x0, y0 = int(kpts_all[t, joint_idx, 0]), int(kpts_all[t, joint_idx, 1])
        x1, y1 = int(kpts_all[t + 1, joint_idx, 0]), int(kpts_all[t + 1, joint_idx, 1])
        if x0 < 1 or x1 < 1:
            continue
        alpha = int(255 * (t - t_start) / max(trail_len, 1))
        cv2.line(canvas, (x0, y0), (x1, y1), (alpha, alpha, 60), 1, cv2.LINE_AA)


def render_progress_bar(canvas: np.ndarray, t: int, T: int):
    """Thanh progress ở dưới cùng."""
    h, w = canvas.shape[:2]
    bar_y = h - 16
    bar_x0, bar_x1 = 10, w - 10
    # Background
    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + 8), (60, 60, 60), -1)
    # Progress
    if T > 1:
        fill_x = int(bar_x0 + (bar_x1 - bar_x0) * t / (T - 1))
        cv2.rectangle(canvas, (bar_x0, bar_y), (fill_x, bar_y + 8), (0, 200, 180), -1)
    cv2.putText(
        canvas, f"{t+1}/{T}", (bar_x0, bar_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
    )


# ──────────────────────────────────────────────────────────────
# PKL visualizer
# ──────────────────────────────────────────────────────────────

def visualize_pkl(pkl_path: str | Path):
    """Load pkl, chạy skeleton animation."""
    pkl_path = Path(pkl_path)
    logger.info(f"Loading pkl: {pkl_path}")

    try:
        data = load_pickle(str(pkl_path))
    except Exception as exc:
        logger.error(f"Không load được pkl: {exc}")
        sys.exit(1)

    # Validate format
    if "annotations" not in data or not data["annotations"]:
        logger.error("pkl không có 'annotations' hoặc rỗng.")
        sys.exit(1)

    ann = data["annotations"][0]
    kp_arr    = ann["keypoint"]        # [M, T, V, 2]
    kp_scores = ann.get("keypoint_score")  # [M, T, V]
    label     = ann.get("label", -1)
    img_shape = ann.get("img_shape", (480, 640))
    sample_id = ann.get("frame_dir", pkl_path.stem)

    M, T, V, C = kp_arr.shape
    logger.info(
        f"  Sample:    {sample_id}\n"
        f"  Shapes:    keypoint={kp_arr.shape}, T={T}, V={V}\n"
        f"  Label:     {label}\n"
        f"  img_shape: {img_shape}"
    )

    # Chỉ lấy person đầu tiên (M=0)
    kpts_mv = kp_arr[0]          # [T, V, 2]
    scores_mv = kp_scores[0] if kp_scores is not None else None  # [T, V]

    # Normalize vào canvas
    kpts_norm = normalize_keypoints(kpts_mv, CANVAS_W, CANVAS_H - 40)

    # ── Animation loop ──
    t = 0
    paused = False
    speed  = 1          # bước nhảy frame
    fps_target = 15.0
    delay_ms   = int(1000 / fps_target)

    logger.info("=" * 55)
    logger.info("  Q/ESC   : thoát")
    logger.info("  SPACE   : pause / resume")
    logger.info("  → / ←  : next / prev frame")
    logger.info("  +/-     : tăng/giảm tốc độ")
    logger.info("  R       : restart")
    logger.info("  S       : screenshot")
    logger.info("=" * 55)

    try:
        while True:
            # ── Canvas ──
            canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

            # Trajectory (spine: joint 11, 12 hips average)
            draw_trajectory(canvas, kpts_norm, t, trail_len=8, joint_idx=0)

            # Skeleton
            s_t = scores_mv[t] if scores_mv is not None else None
            draw_skeleton_frame(canvas, kpts_norm[t], s_t, color_hint=(0, 220, 180))

            # Joint labels (toggle nếu muốn)
            for v in range(min(V, len(COCO_JOINT_NAMES))):
                if s_t is not None and float(s_t[v]) < 0.2:
                    continue
                x, y = int(kpts_norm[t, v, 0]), int(kpts_norm[t, v, 1])
                if x < 1 or y < 1:
                    continue
                cv2.putText(
                    canvas, str(v), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1,
                )

            # Progress bar
            render_progress_bar(canvas, t, T)

            # Info panel
            draw_info_panel(canvas, {
                "Run ADL (Viz)": "",
                "File":    pkl_path.name[:28],
                "T":       f"{t+1}/{T}",
                "V":       str(V),
                "Label":   str(label),
                "Speed":   f"x{speed}",
                "PAUSED" if paused else "PLAYING": "",
            })

            cv2.imshow(
                f"CPose — ADL Skeleton Viz | {sample_id[:40]}  [Q: thoát | SPACE: pause]",
                canvas,
            )
            key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == 81 or key == ord('a'):   # ← / A
                t = max(0, t - 1)
                paused = True
            elif key == 83 or key == ord('d'):   # → / D
                t = min(T - 1, t + 1)
                paused = True
            elif key in (ord('+'), ord('=')):
                speed = min(speed + 1, 8)
            elif key in (ord('-'), ord('_')):
                speed = max(1, speed - 1)
            elif key in (ord('r'), ord('R')):
                t = 0
                paused = False
            elif key in (ord('s'), ord('S')):
                ss_dir  = ensure_dir(ROOT / "data" / "output" / "screenshots")
                ss_path = str(ss_dir / f"adl_{pkl_path.stem}_t{t:04d}.jpg")
                cv2.imwrite(ss_path, canvas)
                logger.info(f"Screenshot: {ss_path}")

            if not paused:
                t = (t + speed) % T   # loop

    finally:
        cv2.destroyAllWindows()
    logger.info("Visualizer closed.")


# ──────────────────────────────────────────────────────────────
# PoseC3D inference mode
# ──────────────────────────────────────────────────────────────

def run_inference(args, cfg):
    """Gọi PoseC3D subprocess (cần MMAction2 đã cài)."""
    from src.action.posec3d import PoseC3DRunner

    runner = PoseC3DRunner(
        mmaction_root=cfg["adl"]["mmaction_root"],
        base_config=cfg["adl"]["base_config"],
        checkpoint=cfg["adl"]["weights"],
        work_dir=cfg["adl"]["work_dir"],
    )
    logger.info(f"Running PoseC3D on: {args.input}")
    result = runner.run_test(args.input)
    if result.returncode != 0:
        logger.error(f"PoseC3D failed (return code {result.returncode})")
        sys.exit(result.returncode)
    logger.info("PoseC3D inference done.")

    # Sau inference, visualize pkl nếu user muốn
    if args.viz_after:
        visualize_pkl(args.input)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def find_latest_pkl(root: Path) -> str | None:
    pkl_dir = root / "data" / "output" / "clips_pkl"
    if not pkl_dir.exists():
        return None
    pkls = sorted(pkl_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pkls[0]) if pkls else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPose — Run ADL: skeleton visualizer + PoseC3D debug"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="MMAction2 pose annotation .pkl (chế độ inference)")
    parser.add_argument("--viz",   type=str, default=None,
                        help="Pkl file để visualize skeleton (không cần MMAction2)")
    parser.add_argument("--viz-latest", action="store_true",
                        help="Tự tìm và viz pkl mới nhất trong data/output/clips_pkl/")
    parser.add_argument("--viz-after",  action="store_true",
                        help="Visualize sau khi chạy inference")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Viz-latest ──
    if args.viz_latest:
        pkl = find_latest_pkl(ROOT)
        if pkl is None:
            logger.error(
                "Không tìm thấy .pkl nào trong data/output/clips_pkl/\n"
                "Chạy run_pipeline.py trước để generate pkl files."
            )
            sys.exit(1)
        logger.info(f"Latest pkl: {pkl}")
        visualize_pkl(pkl)
        return

    # ── Direct viz ──
    if args.viz:
        visualize_pkl(args.viz)
        return

    # ── Inference mode ──
    if args.input:
        cfg = load_pipeline_cfg(Path(args.config), ROOT)
        run_inference(args, cfg)
        return

    # ── Không có argument: hướng dẫn ──
    logger.info("Không có argument. Cách dùng:")
    logger.info("")
    logger.info("  1) Visualize skeleton từ pkl (KHÔNG cần MMAction2):")
    logger.info("       python apps/run_adl.py --viz <path_to_pkl>")
    logger.info("")
    logger.info("  2) Auto-find pkl mới nhất:")
    logger.info("       python apps/run_adl.py --viz-latest")
    logger.info("")
    logger.info("  3) Run PoseC3D inference (cần MMAction2):")
    logger.info("       python apps/run_adl.py --input <path_to_pkl>")
    logger.info("")

    # Nếu có pkl nào trong output, auto-viz
    pkl = find_latest_pkl(ROOT)
    if pkl:
        logger.info(f"Tìm thấy pkl mới nhất: {pkl}")
        logger.info("Tự động visualize...")
        visualize_pkl(pkl)
    else:
        logger.info("Chưa có pkl. Chạy run_pipeline.py trước để tạo pose clips.")


if __name__ == "__main__":
    main()