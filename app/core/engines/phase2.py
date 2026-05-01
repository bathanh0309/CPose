import logging
import cv2
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger("[Engine-Phase2]")

def run_phase2(model, clip_path: Path, output_dir: Path, conf_threshold: float = 0.3) -> Tuple[int, int, Path]:
    """
    Pure Engine for Phase 2: Frame-by-frame person detection.
    Returns: (frames_processed, labels_written, label_file_path)
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open clip: {clip_path}")

    clip_stem = clip_path.stem
    label_file = output_dir / f"{clip_stem}_labels.txt"
    
    frames_count = 0
    labels_count = 0
    
    with label_file.open("w", encoding="utf-8") as f:
        f.write("# frame_id class_id conf x1 y1 x2 y2\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            if results:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    f.write(f"{frames_count} {cls} {conf:.4f} {' '.join(f'{v:.1f}' for v in xyxy)}\n")
                    labels_count += 1
            
            frames_count += 1

    cap.release()
    return frames_count, labels_count, label_file
