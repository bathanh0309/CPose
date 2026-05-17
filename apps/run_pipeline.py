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
import numpy as np

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


def l2_normalize(vec):
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr) + 1e-12
    return arr / norm


def cosine_sim(a, b):
    if a is None or b is None:
        return None
    a = l2_normalize(a)
    b = l2_normalize(b)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def load_face_prototypes(gallery_sources):
    """Load face prototypes from gallery paths (face_*.npy or .pkl with face.prototype).

    Returns dict: person_id -> l2-normalized prototype (np.ndarray).
    """
    prototypes = {}
    if not gallery_sources:
        return prototypes
    import pickle
    for src in gallery_sources or []:
        try:
            p = Path(src)
            if p.is_file():
                if p.suffix.lower() == ".pkl":
                    try:
                        with p.open("rb") as fh:
                            data = pickle.load(fh)
                        face = data.get("face") if isinstance(data, dict) else None
                        proto = face.get("prototype") if isinstance(face, dict) else None
                        if proto is not None:
                            pid = str(data.get("person_id") or p.stem.replace("_embeddings", ""))
                            prototypes[pid] = l2_normalize(proto)
                    except Exception:
                        continue
                continue
            if p.is_dir():
                arrs = []
                for f in sorted(p.glob("face_*.npy")):
                    try:
                        arr = np.load(str(f)).astype(np.float32).reshape(-1)
                        arrs.append(arr)
                    except Exception:
                        continue
                if arrs:
                    proto = np.mean(np.stack(arrs, axis=0), axis=0)
                    proto = proto / (np.linalg.norm(proto) + 1e-12)
                    prototypes[p.name] = proto.astype(np.float32)
                    continue
                # fallback: check for pkl inside dir
                for f in sorted(p.glob("*.pkl")):
                    try:
                        with f.open("rb") as fh:
                            data = pickle.load(fh)
                        face = data.get("face") if isinstance(data, dict) else None
                        proto = face.get("prototype") if isinstance(face, dict) else None
                        if proto is not None:
                            pid = str(data.get("person_id") or f.stem.replace("_embeddings", ""))
                            prototypes[pid] = l2_normalize(proto)
                            break
                    except Exception:
                        continue
        except Exception:
            continue
    return prototypes


def load_face_model(enabled: bool):
    if not enabled:
        return None
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        logger.warning("Face model (insightface) not available.")
        return None
    try:
        app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app
    except Exception as exc:
        logger.warning(f"Init face model failed: {exc}")
        return None


def extract_face_feat(face_model, crop_bgr):
    if face_model is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        faces = face_model.get(crop_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
        proto = getattr(face, "normed_embedding", None)
        if proto is None:
            return None
        arr = np.asarray(proto, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr) + 1e-12
        return (arr / norm).astype(np.float32)
    except Exception:
        return None


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


def assign_unique_reid_matches(
    reid_candidates,
    threshold: float,
    prev_gid_map: dict | None = None,
    thresholds_overrides: dict | None = None,
    prev_hold_map: dict | None = None,
    min_hold_frames: int = 0,
):
    """Greedily assign at most one gallery ID to one bbox per frame.

    Supports candidate tuples in two formats:
      - legacy: (score, tid, person_id)
      - extended: (sort_score, tid, person_id, body_score, face_score)

    thresholds_overrides may include modality-specific keys:
      - 'body', 'face' (base thresholds)
      - 'known_to_known', 'unknown_to_known', 'known_to_unknown' (fallbacks)
      - 'body_known_to_known', 'face_known_to_known', etc. (optional)
    """
    assignments: dict[int, tuple[str, float]] = {}
    used_ids: set[str] = set()
    overrides = thresholds_overrides or {}
    known_to_known = float(overrides.get("known_to_known", threshold))
    unknown_to_known = float(overrides.get("unknown_to_known", threshold))
    known_to_unknown = float(overrides.get("known_to_unknown", threshold))

    body_base = float(overrides.get("body", overrides.get("threshold_body", threshold)))
    face_base = float(overrides.get("face", overrides.get("threshold_face", threshold)))

    body_known_to_known = float(overrides.get("body_known_to_known", overrides.get("known_to_known", body_base)))
    face_known_to_known = float(overrides.get("face_known_to_known", overrides.get("known_to_known", face_base)))
    body_unknown_to_known = float(overrides.get("body_unknown_to_known", overrides.get("unknown_to_known", body_base)))
    face_unknown_to_known = float(overrides.get("face_unknown_to_known", overrides.get("unknown_to_known", face_base)))
    body_known_to_unknown = float(overrides.get("body_known_to_unknown", overrides.get("known_to_unknown", body_base)))
    face_known_to_unknown = float(overrides.get("face_known_to_unknown", overrides.get("known_to_unknown", face_base)))

    # Protective threshold if a track's current gid has not yet been held
    known_to_known_protect = float(overrides.get("known_to_known_protect", min(0.999, known_to_known + 0.15)))

    for item in sorted(reid_candidates, key=lambda item: item[0], reverse=True):
        # Parse candidate tuple
        try:
            if len(item) >= 5:
                sort_score, tid, person_id, body_score, face_score = item
                body_score = None if body_score is None else float(body_score)
                face_score = None if face_score is None else float(face_score)
                score = float(sort_score)
            else:
                score, tid, person_id = item
                body_score = float(score)
                face_score = None
        except Exception:
            continue

        tid_int = int(tid)
        person_id_str = str(person_id)

        prev_gid = None
        if prev_gid_map is not None:
            pg = prev_gid_map.get(tid_int)
            if isinstance(pg, dict):
                prev_gid = pg.get("gid")
            else:
                prev_gid = pg

        # Decide modality to use for thresholding (prefer face when available and strong)
        modality = "body"
        used_score = body_score
        if face_score is not None:
            # prefer face if it is at least as strong as body or above face_base
            if face_score >= face_base or (body_score is None or face_score >= body_score):
                modality = "face"
                used_score = face_score
            else:
                modality = "body"
                used_score = body_score if body_score is not None else face_score

        # Pick applicable threshold depending on transition and modality
        if prev_gid and prev_gid not in {"unknown", "too_small"} and person_id_str not in {"unknown", "too_small"} and person_id_str != prev_gid:
            applicable_threshold = face_known_to_known if modality == "face" else body_known_to_known
            # If the previous gid was held for fewer frames than configured, protect against switching
            try:
                hold_count = 0
                if prev_hold_map is not None:
                    hold_count = int(prev_hold_map.get(tid_int, 0))
            except Exception:
                hold_count = 0
            min_hold = int(min_hold_frames or int(overrides.get("min_hold_frames", 0)))
            if hold_count < min_hold:
                applicable_threshold = max(applicable_threshold, known_to_known_protect)
        elif (prev_gid in {None, "unknown"} and person_id_str not in {"unknown", "too_small"}):
            applicable_threshold = face_unknown_to_known if modality == "face" else body_unknown_to_known
        elif (prev_gid and prev_gid not in {"unknown", "too_small"} and person_id_str in {"unknown"}):
            applicable_threshold = face_known_to_unknown if modality == "face" else body_known_to_unknown
        else:
            applicable_threshold = face_base if modality == "face" else body_base

        applicable_threshold = max(0.0, min(0.999, float(applicable_threshold)))

        if used_score is None or used_score < applicable_threshold:
            continue

        if tid_int in assignments or person_id_str in used_ids:
            continue

        assignments[tid_int] = (person_id_str, float(used_score))
        used_ids.add(person_id_str)

    return assignments


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
            # Try to load face prototypes (face_*.npy or pkl) from the same gallery sources
            try:
                face_prototypes = load_face_prototypes(gallery_sources)
                face_model = load_face_model(bool(face_prototypes))
                if face_prototypes:
                    logger.info(f"Loaded face prototypes: {len(face_prototypes)} persons")
                else:
                    face_prototypes = {}
                    face_model = None
            except Exception as exc:
                logger.warning(f"Face prototypes unavailable: {exc}")
                face_prototypes = {}
                face_model = None

    except Exception as exc:
        reid_warning = f"OSNet ReID unavailable: {exc}"
        logger.warning(reid_warning)
        face_prototypes = {}
        face_model = None

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
    # Mapping local track id -> last assigned global id (e.g., gallery person id or 'unknown')
    track_gid_map: dict[int, str] = {}
    # How many consecutive frames the current gid has been held for each track
    track_gid_hold_count: dict[int, int] = {}
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
                # cleanup reid maps for dead tracks
                try:
                    track_gid_map.pop(tid, None)
                except Exception:
                    pass
                try:
                    track_gid_hold_count.pop(tid, None)
                except Exception:
                    pass
            prev_track_ids = curr_ids

            reid_frame_data: dict[int, dict] = {}
            reid_candidates: list = []
            if osnet_reid is not None:
                for det_reid in detections:
                    tid_reid = int(det_reid.get("track_id", -1))
                    crop_reid, bbox_reid = clip_bbox(frame, det_reid["bbox"])
                    if tid_reid < 0 or crop_reid is None:
                        continue
                    try:
                        x1, y1, x2, y2 = map(float, det_reid["bbox"])
                        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        if area < osnet_reid.min_crop_area:
                            reid_frame_data[tid_reid] = {
                                "gid": "too_small",
                                "score": 0.0,
                                "status": "too_small",
                                "matches": [],
                                "crop": crop_reid,
                                "bbox_clipped": bbox_reid,
                            }
                            continue
                        matches = osnet_reid.get_top_matches(crop_reid, topk=3)
                        reid_frame_data[tid_reid] = {
                            "gid": "unknown",
                            "score": 0.0,
                            "status": "gallery_miss",
                            "matches": matches,
                            "crop": crop_reid,
                            "bbox_clipped": bbox_reid,
                        }
                        # Extract face feature once per detection (if face model available)
                        face_feat = None
                        try:
                            if "face_model" in globals() and face_model is not None:
                                face_feat = extract_face_feat(face_model, crop_reid)
                        except Exception:
                            face_feat = None

                        for person_id, score, _ in matches:
                            # Compute face similarity to gallery prototype if available
                            face_sim = None
                            try:
                                proto = None
                                if "face_prototypes" in globals() and face_feat is not None:
                                    proto = face_prototypes.get(str(person_id))
                                if proto is not None and face_feat is not None:
                                    face_sim = cosine_sim(face_feat, proto)
                            except Exception:
                                face_sim = None
                            # sort by face when available, else by body score
                            sort_score = float(face_sim) if face_sim is not None else float(score)
                            reid_candidates.append((float(sort_score), tid_reid, str(person_id), float(score), None if face_sim is None else float(face_sim)))
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] ReID failed tid={tid_reid}: {exc}", exc_info=True)
                        reid_frame_data[tid_reid] = {
                            "gid": f"track_{tid_reid}",
                            "score": 0.0,
                            "status": "reid_failed",
                            "matches": [],
                            "crop": crop_reid,
                            "bbox_clipped": bbox_reid,
                        }

                # Determine whether to apply transition thresholds
                base_threshold = float(getattr(osnet_reid, "threshold", cfg["reid"].get("threshold", 0.65)))
                use_transitions = bool(cfg.get("reid", {}).get("use_transition_thresholds", True))
                if use_transitions:
                    unique_assignments = assign_unique_reid_matches(
                        reid_candidates,
                        threshold=base_threshold,
                        prev_gid_map=track_gid_map,
                        thresholds_overrides={
                            "known_to_known": cfg["reid"].get("threshold_known_to_known", base_threshold),
                            "unknown_to_known": cfg["reid"].get("threshold_unknown_to_known", base_threshold),
                            "known_to_unknown": cfg["reid"].get("threshold_known_to_unknown", base_threshold),
                            "body": cfg["reid"].get("threshold_body", cfg["reid"].get("threshold", base_threshold)),
                            "face": cfg["reid"].get("threshold_face", cfg["reid"].get("threshold", base_threshold)),
                            "body_known_to_known": cfg["reid"].get("threshold_body_known_to_known", cfg["reid"].get("threshold_known_to_known", base_threshold)),
                            "face_known_to_known": cfg["reid"].get("threshold_face_known_to_known", cfg["reid"].get("threshold_known_to_known", base_threshold)),
                            "body_unknown_to_known": cfg["reid"].get("threshold_body_unknown_to_known", cfg["reid"].get("threshold_unknown_to_known", base_threshold)),
                            "face_unknown_to_known": cfg["reid"].get("threshold_face_unknown_to_known", cfg["reid"].get("threshold_unknown_to_known", base_threshold)),
                            "body_known_to_unknown": cfg["reid"].get("threshold_body_known_to_unknown", cfg["reid"].get("threshold_known_to_unknown", base_threshold)),
                            "face_known_to_unknown": cfg["reid"].get("threshold_face_known_to_unknown", cfg["reid"].get("threshold_known_to_unknown", base_threshold)),
                            "known_to_known_protect": cfg["reid"].get("threshold_known_to_known_protect", None),
                        },
                        prev_hold_map=track_gid_hold_count,
                        min_hold_frames=int(cfg["reid"].get("min_hold_frames", 0)),
                    )
                else:
                    # Simple greedy assignment using base threshold (preferred for stable UI)
                    unique_assignments = assign_unique_reid_matches(
                        reid_candidates,
                        threshold=base_threshold,
                        thresholds_overrides={
                            "body": cfg["reid"].get("threshold_body", cfg["reid"].get("threshold", base_threshold)),
                            "face": cfg["reid"].get("threshold_face", cfg["reid"].get("threshold", base_threshold)),
                            "known_to_known_protect": cfg["reid"].get("threshold_known_to_known_protect", None),
                        },
                        prev_hold_map=track_gid_hold_count,
                        min_hold_frames=int(cfg["reid"].get("min_hold_frames", 0)),
                    )

                # Apply assignments and also reorder the per-track matches so the
                # ReID panel shows the assigned gallery id first (keeps UI in sync)
                for tid_assigned, (gid_assigned, score_assigned) in unique_assignments.items():
                    if tid_assigned in reid_frame_data:
                        info = reid_frame_data[tid_assigned]
                        info["gid"] = gid_assigned
                        info["score"] = score_assigned
                        info["status"] = "gallery_match"
                        matches = info.get("matches") or []
                        # Move assigned gid to front of matches list if present
                        try:
                            idx = next(i for i, (pid, _, _) in enumerate(matches) if str(pid) == str(gid_assigned))
                        except StopIteration:
                            idx = None
                        if idx is not None and idx != 0:
                            m = matches.pop(idx)
                            matches.insert(0, m)
                        info["matches"] = matches

                # Update per-track last assigned global id mapping so next-frame matching
                # can apply per-transition thresholds (ID->ID harder, unknown<->ID easier).
                for tid_upd, info in reid_frame_data.items():
                    try:
                        gid_now = info.get("gid", "unknown")
                        if gid_now is None:
                            gid_now = "unknown"
                        tid_key = int(tid_upd)
                        old_gid = track_gid_map.get(tid_key)
                        if old_gid == gid_now:
                            track_gid_hold_count[tid_key] = int(track_gid_hold_count.get(tid_key, 0)) + 1
                        else:
                            track_gid_hold_count[tid_key] = 1
                        track_gid_map[tid_key] = str(gid_now)
                    except Exception:
                        pass

            # ── 3+4: Body+Face ReID → GlobalID ──────────────────────────────
            for det in detections:
                tid = int(det.get("track_id", -1))
                crop, bbox_clipped = clip_bbox(frame, det["bbox"])

                # ── Face embedding (placeholder — connect ArcFace khi có) ───
                # No ArcFace branch is active in realtime yet.
                # body_only mode: ArcFace is not implemented in realtime yet.

                # ── Assign GlobalID ──────────────────────────────────────────
                if osnet_reid is not None and tid >= 0 and tid in reid_frame_data:
                    info = reid_frame_data[tid]
                    gid = info.get("gid", "unknown")
                    reid_score = float(info.get("score", 0.0))
                    reid_status = info.get("status", "gallery_miss")
                    weights = {"mode": "body_only"}
                    if info.get("matches"):
                        last_matches = info["matches"]
                    if info.get("crop") is not None:
                        last_query_crop = info["crop"].copy()
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
