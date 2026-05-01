import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from collections import defaultdict, deque

# Import utilities from the legacy file to reuse logic
from app.utils.pose_utils import draw_skeleton, rule_based_adl
from app.core.adl_skeleton import normalize_skeleton
from app.core.adl_model import ADLModelWrapper
from app.core.recognizer_utils import (
    SequentialTracker, 
    PoseTemporalSmoothing, 
    ADLTemporalSmoothing
)



logger = logging.getLogger("[Engine-Phase3]")

def run_phase3(
    model, 
    adl_model: ADLModelWrapper,
    clip_path: Path, 
    output_dir: Path, 
    config: Dict[str, Any],
    save_overlay: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event=None,
) -> Dict[str, Any]:
    """
    Pure Engine for Phase 3: Pose Extraction + ADL Recognition.
    Returns: Dict containing summary of results.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open clip: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    clip_output_dir = output_dir / clip_path.stem
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = output_dir.parent / "output_process"
    processed_clip_dir = processed_root / clip_path.stem
    processed_clip_dir.mkdir(parents=True, exist_ok=True)
    processed_video_path = processed_clip_dir / f"{clip_path.stem}_processed.mp4"

    keypoints_file = clip_output_dir / f"{clip_path.stem}_keypoints.txt"
    adl_file = clip_output_dir / f"{clip_path.stem}_adl.txt"

    # Settings from config
    window_size = max(1, int(config.get("window_size", 30)))
    kp_min_conf = float(config.get("keypoint_conf_min", 0.30))
    conf_threshold = float(config.get("conf_threshold", 0.45))
    person_class_id = int(config.get("person_class_id", 0))
    progress_every = max(1, int(config.get("progress_every", 3)))
    save_overlay = bool(config.get("save_overlay", True))

    # Internal state managers
    tracker = SequentialTracker(
        iou_threshold=float(config.get("track_iou_threshold", 0.20)),
        max_missed_frames=int(config.get("track_max_missed_frames", 15)),
        center_distance_ratio=float(config.get("track_center_distance_ratio", 0.18))
    )
    pose_smoother = PoseTemporalSmoothing(pose_ttl=5)
    adl_smoother = ADLTemporalSmoothing(hold_frames=8, switch_margin=0.08)
    person_windows = defaultdict(lambda: deque(maxlen=window_size))
    latest_adl_by_track: dict[int, tuple[str, float]] = {}
    
    keypoints_count = 0
    adl_count = 0
    frame_id = 0
    processed_writer = None
    processed_writer_failed = False

    with keypoints_file.open("w", encoding="utf-8") as kp_h, adl_file.open("w", encoding="utf-8") as adl_h:
        kp_h.write("# frame_id track_id kps...\n")
        adl_h.write("# frame_id track_id adl_label confidence\n")

        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("Phase 3 stop requested for %s", clip_path.name)
                break

            ret, frame = cap.read()
            if not ret:
                break

            if save_overlay and processed_writer is None and not processed_writer_failed and frame is not None:
                height, width = frame.shape[:2]
                if width > 0 and height > 0:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    processed_writer = cv2.VideoWriter(str(processed_video_path), fourcc, fps, (width, height))
                    if not processed_writer.isOpened():
                        logger.warning("Unable to open processed video writer for %s", processed_video_path)
                        processed_writer.release()
                        processed_writer = None
                        processed_writer_failed = True
            
            # 1. Detection
            results = model.predict(frame, classes=[person_class_id], conf=conf_threshold, verbose=False)
            detections = []
            if results and results[0].boxes:
                keypoints_xy = None
                keypoints_conf = None
                if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                    try:
                        keypoints_xy = results[0].keypoints.xy.cpu().numpy()
                        keypoints_conf = results[0].keypoints.conf.cpu().numpy()
                    except Exception:
                        keypoints_xy = None
                        keypoints_conf = None

                for idx, box in enumerate(results[0].boxes):
                    kp_xy = keypoints_xy[idx] if keypoints_xy is not None and idx < len(keypoints_xy) else np.zeros((17, 2))
                    kp_conf = keypoints_conf[idx] if keypoints_conf is not None and idx < len(keypoints_conf) else np.zeros(17)
                    detections.append({
                        "bbox": box.xyxy[0].cpu().numpy(),
                        "keypoints_xy": kp_xy,
                        "keypoints_conf": kp_conf,
                        "detection_conf": float(box.conf[0])
                    })

            # 2. Tracking
            tracked_people, expired_track_ids = tracker.update(detections, frame.shape[:2], frame_id)
            
            # 3. Processing each person
            processed_frame = frame.copy()
            best_adl = "unknown"
            best_conf = 0.0
            latest_adl_by_track.clear()

            for person in tracked_people:
                tid = person.track_id
                
                # Pose smoothing
                combined_kp = np.column_stack([person.keypoints_xy, person.keypoints_conf])
                smoothed = pose_smoother.merge_pose(tid, combined_kp)
                p_xy, p_conf = smoothed[:, :2], smoothed[:, 2]

                # ADL Window logic
                window = person_windows[tid]
                window.append((p_xy, p_conf))
                if len(window) == window_size:
                    if adl_model is not None:
                        label, conf = adl_model.infer_sequence(
                            np.stack([xy for xy, c in window]),
                            np.stack([c for xy, c in window]),
                        )
                    else:
                        label, conf = rule_based_adl(list(window), config)

                    s_label, s_conf = adl_smoother.smooth_adl(tid, label, conf)
                    latest_adl_by_track[tid] = (s_label, s_conf)
                    adl_h.write(f"{frame_id} {tid} {s_label} {s_conf:.2f}\n")
                    adl_count += 1

                    if s_conf > best_conf:
                        best_conf = s_conf
                        best_adl = s_label
                else:
                    latest_adl_by_track.setdefault(tid, ("unknown", 0.0))

                x1, y1, x2, y2 = [int(round(v)) for v in person.bbox]
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                processed_frame = draw_skeleton(processed_frame, p_xy, p_conf, min_conf=kp_min_conf)

                track_label = f"ID {tid}"
                adl_label, adl_score = latest_adl_by_track.get(tid, ("unknown", 0.0))
                if adl_label and adl_label != "unknown":
                    track_label += f" | {adl_label.upper()}"
                if adl_score:
                    track_label += f" {int(round(adl_score * 100))}%"
                text_y = max(20, y1 - 8)
                cv2.putText(
                    processed_frame,
                    track_label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Write Keypoints
                flat_kp = " ".join([f"{v:.1f}" for row in smoothed for v in row])
                kp_h.write(f"{frame_id} {tid} {flat_kp}\n")
                keypoints_count += 1

            # Trigger progress callback 
            if processed_writer is not None:
                processed_writer.write(processed_frame)

            if progress_callback and (frame_id % progress_every == 0 or frame_id >= total_frames - 1):
                progress_callback({
                    "frame_id": frame_id,
                    "total_frames": total_frames,
                    "adl": best_adl,
                    "conf": best_conf,
                    "original": frame,
                    "processed": processed_frame,
                })
            
            # Cleanup smoothing memory
            for etid in expired_track_ids:
                pose_smoother.expire_track(etid)
                adl_smoother.expire_track(etid)
                person_windows.pop(etid, None)

            frame_id += 1

    cap.release()
    if processed_writer is not None:
        processed_writer.release()
    return {
        "clip_stem": clip_path.stem,
        "frames_processed": frame_id,
        "keypoints_written": keypoints_count,
        "adl_events": adl_count,
        "output_dir": str(clip_output_dir),
        "processed_video_path": str(processed_video_path) if processed_video_path.exists() else None,
    }
