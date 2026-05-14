import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_pipeline_cfg

DEFAULT_SOURCE = ROOT / "data" / "input" / "cam1_2026-01-29_16-26-25.mp4"
OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug YOLO pose detection")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="image/video path or webcam index",
    )
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def result_to_dict(result):
    items = []
    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or keypoints is None:
        return items

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    kp_xy = keypoints.xy.cpu().numpy()
    kp_conf = None
    if keypoints.conf is not None:
        kp_conf = keypoints.conf.cpu().numpy()

    for i in range(len(xyxy)):
        items.append({
            "bbox": xyxy[i].tolist(),
            "score": float(confs[i]),
            "class_id": int(cls_ids[i]),
            "track_id": -1,
            "keypoints": kp_xy[i].tolist(),
            "keypoint_scores": kp_conf[i].tolist() if kp_conf is not None else None,
        })
    return items


def main():
    args = parse_args()

    from ultralytics import YOLO

    cfg = load_pipeline_cfg(Path(ROOT / "configs" / "system" / "pipeline.yaml"), ROOT)
    model = YOLO(str(cfg["pose"]["weights"]))

    source = int(args.source) if args.source.isdigit() else args.source
    results = model.predict(
        source=source,
        conf=args.conf,
        save=args.save_vis,
        project=str(OUTPUT_DIR),
        name="pose_vis",
        verbose=False,
        stream=True,
    )

    all_results = []
    for idx, result in enumerate(results):
        all_results.append({
            "frame_index": idx,
            "detections": result_to_dict(result),
        })

    out_json = OUTPUT_DIR / "pose_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved pose results to: {out_json}")


if __name__ == "__main__":
    main()
