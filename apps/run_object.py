import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, find_default_video_source, get_video_meta, open_video_source, safe_imshow
from src.utils.vis import FPSCounter, draw_info_panel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run custom object detector")
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


def draw_object(frame, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 255), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 180, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    obj_cfg = cfg["object"]
    weights = Path(obj_cfg["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Object detector weights not found: {weights}. Train a custom model and set [object].weights.")

    model = YOLO(str(weights))
    source = args.source or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Put a video at data/sample.mp4 or data/input/, or pass --source.")

    logger.info(f"Opening video source: {source}")
    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    writer = None
    if args.save_video:
        out_path = Path(args.output) if args.output else resolve_output_path(
            cfg["system"]["vis_dir"],
            make_video_output_name("object", args.camera_id),
        )
        writer = create_video_writer(out_path, fps, width, height)
        logger.info(f"Saving video to: {out_path}")

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

            results = model.predict(
                source=frame,
                conf=float(obj_cfg.get("conf", 0.45)),
                iou=float(obj_cfg.get("iou", 0.5)),
                device=cfg["system"]["device"],
                verbose=False,
            )
            result = results[0]
            count = 0
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    name = result.names.get(cls_id, str(cls_id))
                    score = float(box.conf.item())
                    count += 1
                    draw_object(frame, box.xyxy[0].cpu().numpy().tolist(), f"{name} {score:.2f}")

            draw_info_panel(frame, {
                "Module": "Custom Object Detector",
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Objects": count,
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_counter.tick():.1f}",
            })

            if writer is not None:
                writer.write(frame)
            if show:
                key = safe_imshow("CPose - Object Detector", frame)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
