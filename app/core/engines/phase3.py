import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from collections import defaultdict, deque

# Import utilities from the legacy file to reuse logic
from app.utils.pose_utils import draw_skeleton, rule_based_adl
from cpose.core.adl.skeleton_norm import normalize_skeleton
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
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Pure Engine for Phase 3: Pose Extraction + ADL Recognition.
    Returns: Dict containing summary of results.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open clip: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    clip_output_dir = output_dir / clip_path.stem
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    keypoints_file = clip_output_dir / f"{clip_path.stem}_keypoints.txt"
    adl_file = clip_output_dir / f"{clip_path.stem}_adl.txt"

    # Settings from config
    window_size = max(1, int(config.get("window_size", 30)))
    kp_min_conf = float(config.get("keypoint_conf_min", 0.30))
    conf_threshold = float(config.get("conf_threshold", 0.45))
    person_class_id = int(config.get("person_class_id", 0))

    # Internal state managers
    tracker = SequentialTracker(
        iou_threshold=float(config.get("track_iou_threshold", 0.20)),
        max_missed_frames=int(config.get("track_max_missed_frames", 15)),
        center_distance_ratio=float(config.get("track_center_distance_ratio", 0.18))
    )
    pose_smoother = PoseTemporalSmoothing(pose_ttl=5)
    adl_smoother = ADLTemporalSmoothing(hold_frames=8, switch_margin=0.08)
    person_windows = defaultdict(lambda: deque(maxlen=window_size))
    
    keypoints_count = 0
    adl_count = 0
    frame_id = 0

    with keypoints_file.open("w", encoding="utf-8") as kp_h, adl_file.open("w", encoding="utf-8") as adl_h:
        kp_h.write("# frame_id track_id kps...\n")
        adl_h.write("# frame_id track_id adl_label confidence\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Detection
            results = model.predict(frame, classes=[person_class_id], conf=conf_threshold, verbose=False)
            detections = []
            if results and results[0].boxes:
                for box in results[0].boxes:
                    detections.append({
                        "bbox": box.xyxy[0].cpu().numpy(),
                        "keypoints_xy": results[0].keypoints.xy[0].cpu().numpy() if hasattr(results[0], 'keypoints') else np.zeros((17, 2)),
                        "keypoints_conf": results[0].keypoints.conf[0].cpu().numpy() if hasattr(results[0], 'keypoints') else np.zeros(17),
                        "detection_conf": float(box.conf[0])
                    })

            # 2. Tracking
            tracked_people, expired_track_ids = tracker.update(detections, frame.shape[:2], frame_id)
            
            # 3. Processing each person
            for person in tracked_people:
                tid = person.track_id
                
                # Pose smoothing
                combined_kp = np.column_stack([person.keypoints_xy, person.keypoints_conf])
                smoothed = pose_smoother.merge_pose(tid, combined_kp)
                p_xy, p_conf = smoothed[:, :2], smoothed[:, 2]

                # Write Keypoints
                flat_kp = " ".join([f"{v:.1f}" for row in smoothed for v in row])
                kp_h.write(f"{frame_id} {tid} {flat_kp}\n")
                keypoints_count += 1

                # ADL Window logic
                window = person_windows[tid]
                window.append((p_xy, p_conf))
                if len(window) == window_size:
                    # Prepare sequence for GCN
                    xy_seq = np.stack([xy for xy, conf in window], axis=0) # (T, V, 2)
                    conf_seq = np.stack([conf for xy, conf in window], axis=0) # (T, V)
                    
                    if adl_model is not None:
                        label, conf = adl_model.infer_sequence(xy_seq, conf_seq)
                    else:
                        label, conf = rule_based_adl(list(window), config)
                        
                    s_label, s_conf = adl_smoother.smooth_adl(tid, label, conf)
                    adl_h.write(f"{frame_id} {tid} {s_label} {s_conf:.2f}\n")
                    adl_count += 1

                    # Trigger progress callback for one person (main track)
                    if progress_callback and frame_id % 10 == 0:
                        progress_callback({
                            "frame_id": frame_id,
                            "total_frames": total_frames,
                            "adl": s_label,
                            "conf": s_conf,
                            "frame": frame # Pass raw frame for snapshot extraction if needed
                        })
            
            # Cleanup smoothing memory
            for etid in expired_track_ids:
                pose_smoother.expire_track(etid)
                adl_smoother.expire_track(etid)
                person_windows.pop(etid, None)

            frame_id += 1

    cap.release()
    return {
        "clip_stem": clip_path.stem,
        "frames_processed": frame_id,
        "keypoints_written": keypoints_count,
        "adl_events": adl_count,
        "output_dir": str(clip_output_dir)
    }
