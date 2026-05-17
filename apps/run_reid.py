import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.pedestrian_yolo import PedestrianYoloTracker
from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.config import get_module_source, load_pipeline_cfg
from src.utils.logger import get_logger, log_frame_metrics
from src.utils.naming import make_video_output_name, resolve_output_path
from src.utils.video import create_video_writer, destroy_all_windows, find_default_video_source, get_video_meta, open_video_source, safe_imshow, toggle_video_recording
from src.utils.vis import FPSCounter, draw_detection, draw_info_panel, draw_reid_panel

logger = get_logger(__name__)

BODY_ONLY_WEIGHTS = {"face_pct": None, "body_pct": 100, "sim_face": None, "sim_body": None}


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11 + ByteTrack + OSNet-x0.25 ReID visualization")
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


def clipped_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-12)


def cosine_sim(a, b):
    if a is None or b is None:
        return None
    a = l2_normalize(a)
    b = l2_normalize(b)
    if a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def load_face_prototypes(gallery_sources):
    prototypes = {}
    for source in gallery_sources or []:
        path = Path(source)
        if not path.exists() or path.suffix.lower() != ".pkl":
            continue
        try:
            with path.open("rb") as file:
                data = pickle.load(file)
        except Exception as exc:
            logger.warning(f"Cannot read face prototype from {path}: {exc}")
            continue
        face = data.get("face") if isinstance(data, dict) else None
        proto = face.get("prototype") if isinstance(face, dict) else None
        if proto is None:
            continue
        pid = str(data.get("person_id") or path.stem.replace("_embeddings", ""))
        prototypes[pid] = l2_normalize(proto)
    return prototypes


def load_face_model(enabled: bool):
    if not enabled:
        return None
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.warning("Face contribution unavailable: insightface is not installed.")
        return None

    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def extract_face_feat(face_model, crop_bgr):
    if face_model is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    faces = face_model.get(crop_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    return l2_normalize(face.normed_embedding)


def modality_weights(sim_face, sim_body):
    if sim_body is None:
        return {"face_pct": None, "body_pct": 0, "sim_face": sim_face, "sim_body": sim_body}
    if sim_face is None:
        return {**BODY_ONLY_WEIGHTS, "sim_body": sim_body}

    face_strength = max(0.0, float(sim_face))
    body_strength = max(0.0, float(sim_body))
    total = face_strength + body_strength
    if total <= 1e-12:
        return {"face_pct": 0, "body_pct": 0, "sim_face": sim_face, "sim_body": sim_body}

    face_pct = int(round(100.0 * face_strength / total))
    body_pct = max(0, 100 - face_pct)
    return {"face_pct": face_pct, "body_pct": body_pct, "sim_face": sim_face, "sim_body": sim_body}


def modality_text(weights: dict) -> str:
    face_pct = weights.get("face_pct")
    body_pct = weights.get("body_pct")
    face_text = "N/A" if face_pct is None else f"{int(face_pct)}%"
    body_text = "N/A" if body_pct is None else f"{int(body_pct)}%"
    return f"F={face_text} B={body_text}"


def reid_bbox_label(track_id: int, gid: str, score: float, status: str, weights: dict) -> str:
    weight_text = modality_text(weights)
    if gid and gid not in {"unknown", "too_small"} and not str(gid).startswith("track_"):
        return f"ID={gid} {weight_text} s={score:.2f}"
    if gid == "too_small":
        return f"track={track_id} too_small {weight_text}"
    return f"track={track_id} UNK {weight_text} s={score:.2f} {status}"


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    if cfg["tracking"].get("model_type") == "pedestrian":
        tracking_weights = cfg["tracking"].get("weights", ROOT / "models/yolo11n.pt")
        if not Path(tracking_weights).exists() and cfg["tracking"].get("fallback_weights"):
            tracking_weights = cfg["tracking"]["fallback_weights"]
        detector = PedestrianYoloTracker(
            weights=tracking_weights,
            conf=cfg["tracking"].get("conf", 0.40),
            iou=cfg["tracking"].get("iou", 0.5),
            tracker=cfg["tracking"]["tracker_yaml"],
            device=cfg["system"]["device"],
            classes=cfg["tracking"].get("classes", [0]),
            tracking_cfg=cfg["tracking"],
        )
        module_name = "YOLO+ByteTrack+OSNet"
    else:
        detector = YoloPoseTracker(
            cfg["pose"]["weights"],
            cfg["pose"]["conf"],
            cfg["pose"]["iou"],
            cfg["tracking"]["tracker_yaml"],
            cfg["system"]["device"],
            tracking_cfg=cfg["tracking"],
        )
        module_name = "YOLO-Pose+ByteTrack+OSNet"
    extractor = None
    reid_warning = None
    gallery_sources = []
    try:
        from src.reid.osnet_reid import OSNetReID

        reid_weights = cfg["reid"]["weights"]
        if not Path(reid_weights).exists() and cfg["reid"].get("fallback_weights"):
            reid_weights = cfg["reid"]["fallback_weights"]
        extractor = OSNetReID(
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
            or []
        )
        loaded = extractor.load_gallery_embeddings(gallery_sources, cfg["reid"].get("id_aliases"))
        if loaded <= 0:
            reid_warning = "OSNet gallery empty"
    except Exception as exc:
        reid_warning = f"OSNet unavailable: {exc}"
        logger.warning(reid_warning)

    face_prototypes = load_face_prototypes(gallery_sources)
    face_model = load_face_model(bool(face_prototypes))

    source = args.source or get_module_source(cfg, "reid") or find_default_video_source(ROOT)
    if source is None:
        raise RuntimeError("No video source found. Set sources.reid or pass --source.")

    logger.info(f"Opening video source: {source}")
    show = not args.no_show
    cap, _ = open_video_source(source)
    width, height, fps, total = get_video_meta(cap)
    writer = None
    panel_w = 220
    out_path = Path(args.output) if args.output else resolve_output_path(
        cfg["system"]["vis_dir"],
        make_video_output_name("reid", args.camera_id),
    )
    if args.save_video:
        writer = create_video_writer(out_path, fps, width + panel_w, height)
        logger.info(f"Recording started: {out_path}")

    if not cfg.get("output", {}).get("save_json", False):
        logger.info("ReID JSON disabled by config")

    last_crop = None
    last_matches = []
    gid_cache = {}
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

            try:
                detections, _ = detector.infer(frame, persist=True)
            except Exception as exc:
                logger.warning(f"[frame {frame_idx}] pose tracking failed: {exc}", exc_info=True)
                continue

            last_gid = "none"
            last_score = 0.0
            last_status = "no_person"
            last_face_pct = None
            last_body_pct = 0
            last_weights = {"face_pct": None, "body_pct": 0, "sim_face": None, "sim_body": None}
            for det in detections:
                tid = det.get("track_id", -1)
                crop = clipped_crop(frame, det["bbox"])
                if tid < 0 or crop is None:
                    draw_detection(frame, det, label=f"track={tid}")
                    continue
                if extractor is not None:
                    try:
                        cached = gid_cache.get(tid)
                        if cached and frame_idx % int(cfg["reid"]["reid_interval"]) != 0:
                            gid, score, status = cached["gid"], cached["score"], "cache_hit"
                            weights = cached.get("weights", BODY_ONLY_WEIGHTS)
                        else:
                                x1, y1, x2, y2 = map(float, det["bbox"])
                                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                                # Use top matches and modality thresholds (configurable)
                                matches = extractor.get_top_matches(crop, topk=3)
                                best_pid = None
                                best_body_sim = None
                                if matches:
                                    best_pid, best_body_sim, _ = matches[0]
                                    best_body_sim = float(best_body_sim)

                                face_feat = extract_face_feat(face_model, crop)
                                sim_face = None
                                if face_feat is not None and best_pid is not None:
                                    proto = face_prototypes.get(str(best_pid))
                                    if proto is not None:
                                        sim_face = cosine_sim(face_feat, proto)

                                body_thresh = float(cfg["reid"].get("threshold_body", cfg["reid"].get("threshold", extractor.threshold)))
                                face_thresh = float(cfg["reid"].get("threshold_face", cfg["reid"].get("threshold", extractor.threshold)))

                                # Prefer face match when available and above face_thresh
                                if sim_face is not None and sim_face >= face_thresh:
                                    gid = str(best_pid)
                                    score = float(sim_face)
                                    status = "gallery_match_face"
                                elif best_body_sim is not None and best_body_sim >= body_thresh:
                                    gid = str(best_pid)
                                    score = float(best_body_sim)
                                    status = "gallery_match"
                                else:
                                    gid = "unknown"
                                    score = float(best_body_sim) if best_body_sim is not None else 0.0
                                    status = "gallery_miss"

                                sim_body = best_body_sim
                                weights = modality_weights(sim_face, sim_body)
                                gid_cache[tid] = {
                                    "gid": gid,
                                    "score": score,
                                    "frame_idx": frame_idx,
                                    "weights": weights,
                                }
                        last_matches = extractor.get_top_matches(crop, topk=3)
                        last_crop = crop.copy()
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID failed for track={tid}: {exc}")
                        gid, score, status = f"track_{tid}", 0.0, "reid_failed"
                        weights = {"face_pct": None, "body_pct": 0, "sim_face": None, "sim_body": None}
                        last_matches = []
                else:
                    gid, score, status = f"track_{tid}", 0.0, "reid_unavailable"
                    weights = {"face_pct": None, "body_pct": 0, "sim_face": None, "sim_body": None}
                    last_crop = crop.copy()
                    last_matches = []
                last_gid = gid
                last_score = score
                last_status = status
                last_weights = weights
                last_face_pct = weights.get("face_pct")
                last_body_pct = weights.get("body_pct")
                draw_detection(frame, det, label=reid_bbox_label(tid, gid, score, status, weights))

            fps_value = fps_counter.tick()
            gallery_size = len(extractor.gallery) if extractor is not None else 0
            metrics_interval = int(cfg.get("ui", {}).get("metrics_interval_frames", 5))
            info = {
                "Module": module_name,
                "Camera": args.camera_id,
                "Frame": f"{frame_idx}/{total}" if total else frame_idx,
                "Persons": len(detections),
                "Gallery": gallery_size,
                "Face Weight": "N/A" if last_face_pct is None else f"{last_face_pct}%",
                "Body Weight": "N/A" if last_body_pct is None else f"{last_body_pct}%",
                "Face Sim": "N/A" if last_weights.get("sim_face") is None else f"{last_weights['sim_face']:.2f}",
                "Body Sim": "N/A" if last_weights.get("sim_body") is None else f"{last_weights['sim_body']:.2f}",
                "Device": cfg["system"]["device"],
                "FPS": f"{fps_value:.1f}",
            }
            log_frame_metrics(
                logger,
                "reid",
                args.camera_id,
                frame_idx,
                fps_value,
                interval=metrics_interval,
                persons=len(detections),
                gallery_size=gallery_size,
                last_gid=last_gid,
                last_score=f"{float(last_score):.2f}",
                last_status=last_status,
                face_weight="N/A" if last_face_pct is None else f"{last_face_pct}%",
                body_weight="N/A" if last_body_pct is None else f"{last_body_pct}%",
                face_sim="N/A" if last_weights.get("sim_face") is None else f"{last_weights['sim_face']:.2f}",
                body_sim="N/A" if last_weights.get("sim_body") is None else f"{last_weights['sim_body']:.2f}",
            )
            if reid_warning:
                info["Warning"] = reid_warning[:48]
            elif extractor is not None and not extractor.gallery:
                info["Warning"] = "ReID gallery empty"
            draw_info_panel(frame, info)
            display = draw_reid_panel(frame, last_crop, last_matches, panel_w=panel_w)

            if writer is not None:
                writer.write(display)
            if show:
                key = safe_imshow("CPose - ReID", display)
                if key in (ord("g"), ord("G")):
                    writer = toggle_video_recording(writer, out_path, fps, width + panel_w, height, logger)
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        destroy_all_windows()


if __name__ == "__main__":
    main()
