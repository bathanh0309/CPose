import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
import logging
from skeleton_norm import normalize_skeleton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[DataPrep]")

def parse_keypoints_txt(txt_path: Path) -> Dict[int, List[np.ndarray]]:
    """
    Parse a CPose _keypoints.txt file.
    Returns: { track_id: [ (17, 3), (17, 3), ... ] }
    """
    tracks = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            kps_data = [float(x) for x in parts[2:]]
            
            # Reshape to (17, 3)
            kps = np.array(kps_data).reshape(-1, 3)
            
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(kps)
            
    return tracks

def convert_to_gcn_format(input_dir: Path, output_pkl: Path, window_size: int = 30):
    """
    Scan for all _keypoints.txt and convert to GCN training format (pkl/npy).
    """
    all_data = [] # List of { 'keypoint': (C, T, V, M), 'label': int }
    
    # In a real scenario, you'd have folders corresponding to labels
    # Here we assume we might need to parse _adl.txt for labels or use folder names
    
    kp_files = list(input_dir.glob("**/*_keypoints.txt"))
    logger.info(f"Found {len(kp_files)} keypoints files.")
    
    for kp_file in kp_files:
        logger.info(f"Processing {kp_file.name}...")
        tracks = parse_keypoints_txt(kp_file)
        
        for tid, sequence in tracks.items():
            if len(sequence) < window_size:
                continue
                
            # Sliding window over the track sequence
            for i in range(0, len(sequence) - window_size + 1, window_size // 2):
                window = sequence[i:i+window_size]
                feat = np.stack(window, axis=0) # (T, V, 3)
                
                # Normalize exactly like in inference
                feat_norm = normalize_skeleton(feat)
                
                # Reshape to (C, T, V, M) - typically what CTR-GCN wants
                # C=3, T=window_size, V=17, M=1
                data = np.transpose(feat_norm, (2, 0, 1)) # (3, T, 17)
                data = data[..., np.newaxis] # (3, T, 17, 1)
                
                all_data.append({
                    'keypoint': data.astype(np.float32),
                    'label': 0, # TODO: Map label from _adl.txt or folder
                    'sample_name': f"{kp_file.stem}_t{tid}_f{i}"
                })

    with open(output_pkl, 'wb') as f:
        pickle.dump(all_data, f)
    
    logger.info(f"Saved {len(all_data)} samples to {output_pkl}")

if __name__ == "__main__":
    # Example usage
    # convert_to_gcn_format(Path("data/output_pose"), Path("training_data.pkl"))
    pass
