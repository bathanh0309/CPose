import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reid.body_extractor import BodyExtractor
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.video import find_default_video_source, get_video_meta, open_video_source, safe_imshow

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build body embedding gallery from a sample video")
    parser.add_argument("--video", default=None, type=str, help="Sample video path. Defaults to pipeline.yaml system.default_source.")
    parser.add_argument("--person-id", default="person01", type=str, help="Output person id folder, e.g. APhu.")
    parser.add_argument("--output-dir", default="data/body", type=str)
    parser.add_argument("--config", default=str(ROOT / "configs/system/pipeline.yaml"))
    parser.add_argument("--max-frames", default=500, type=int)
    parser.add_argument("--sample-every", default=10, type=int)
    parser.add_argument("--use-detector", action="store_true", default=True)
    parser.add_argument("--no-detector", dest="use_detector", action="store_false")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def save_meta(person_dir: Path, meta: dict):
    person_dir.mkdir(parents=True, exist_ok=True)
    meta_path = person_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


def draw_status(frame, person_id, frame_idx, saved_count, sample_every):
    cv2.rectangle(frame, (8, 8), (560, 112), (255, 255, 255), -1)
    cv2.rectangle(frame, (8, 8), (560, 112), (40, 40, 40), 1)
    lines = [
        "Module: Build Body Gallery",
        f"Person: {person_id}  |  Frame: {frame_idx}  |  Saved: {saved_count}",
        f"Sample every: {sample_every} frames  |  Q/Esc: quit",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (18, 36 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)


def draw_bbox(frame, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 80), 2)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 180, 80), 2, cv2.LINE_AA)


def create_extractor(cfg):
    from src.reid.fast_reid import FastReIDExtractor

    return FastReIDExtractor(
        config=cfg["reid"]["fastreid_config"],
        weights_path=cfg["reid"]["weights"],
        device=cfg["system"]["device"],
        output_dir=cfg["reid"].get("output_dir"),
        fastreid_root=cfg["reid"].get("fastreid_root"),
    )


def create_detector(cfg):
    from src.detectors.yolo_pose import YoloPoseTracker

    return YoloPoseTracker(
        weights=cfg["pose"]["weights"],
        conf=cfg["pose"]["conf"],
        iou=cfg["pose"]["iou"],
        tracker=cfg["tracking"]["tracker_yaml"],
        device=cfg["system"]["device"],
        tracking_cfg=cfg["tracking"],
    )


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    video_path = args.video or cfg["system"].get("default_source") or find_default_video_source(ROOT)
    if video_path is None:
        raise RuntimeError("No video source found. Pass --video or set system.default_source in pipeline.yaml.")

    try:
        extractor = create_extractor(cfg)
    except Exception as exc:
        logger.error(f"FastReID unavailable: {exc}")
        sys.exit(1)

    detector = None
    if args.use_detector:
        try:
            detector = create_detector(cfg)
            logger.info("YOLO detector enabled; body crops will be extracted from detected persons.")
        except Exception as exc:
            logger.warning(f"Detector unavailable; using full-frame crops instead: {exc}")

    output_dir = ROOT / args.output_dir
    person_dir = output_dir / args.person_id
    body_extractor = BodyExtractor(extractor)
    cap, _ = open_video_source(video_path)
    width, height, fps, total = get_video_meta(cap)
    show = not args.no_show
    saved_paths: list[str] = []
    scores: list[float] = []
    frame_idx = -1
    started_at = time.time()

    logger.info(f"Opening video source: {video_path}")
    logger.info(f"Saving body embeddings to: {person_dir}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            display = frame.copy()
            sampled = args.sample_every <= 1 or frame_idx % args.sample_every == 0
            detections_count = 0
            saved_this_frame = 0

            if detector is not None:
                try:
                    detections, _ = detector.infer(frame, persist=None)
                except Exception as exc:
                    logger.warning(f"[frame {frame_idx}] detector failed: {exc}")
                    detections = []

                detections_count = len(detections)
                for det in detections:
                    draw_bbox(display, det["bbox"], f"person {float(det.get('score', 0.0)):.2f}")
                    if not sampled:
                        continue
                    feat = body_extractor.extract_from_bbox(frame, det["bbox"])
                    if feat is None:
                        continue
                    person_dir.mkdir(parents=True, exist_ok=True)
                    out_path = person_dir / f"emb_{len(saved_paths):02d}.npy"
                    np.save(str(out_path), feat)
                    saved_paths.append(str(out_path))
                    scores.append(float(det.get("score", 0.0)))
                    saved_this_frame += 1
            elif sampled:
                feat = body_extractor.extract_from_crop(frame)
                if feat is not None:
                    person_dir.mkdir(parents=True, exist_ok=True)
                    out_path = person_dir / f"emb_{len(saved_paths):02d}.npy"
                    np.save(str(out_path), feat)
                    saved_paths.append(str(out_path))
                    saved_this_frame += 1

            log_frame_metrics(
                logger,
                "body_gallery",
                args.person_id,
                frame_idx,
                fps,
                interval=int(cfg.get("ui", {}).get("metrics_interval_frames", 5)),
                detections=detections_count,
                saved_total=len(saved_paths),
                saved_this_frame=saved_this_frame,
                sampled=sampled,
            )

            draw_status(display, args.person_id, frame_idx, len(saved_paths), args.sample_every)
            if show:
                key = safe_imshow("CPose - Build Body Gallery", display)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    meta = {
        "person_id": args.person_id,
        "source_video": str(video_path),
        "video_width": width,
        "video_height": height,
        "video_fps": fps,
        "video_total_frames": total,
        "total_embeddings": len(saved_paths),
        "avg_detection_score": float(np.mean(scores)) if scores else 0.0,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started_at, 2),
        "note": "body embeddings via FastReID",
    }
    save_meta(person_dir, meta)
    logger.info(f"Saved {len(saved_paths)} body embeddings at: {person_dir}")
    logger.info("Run run_reid.py or run_pipeline.py to use the updated body gallery.")


if __name__ == "__main__":
    main()
