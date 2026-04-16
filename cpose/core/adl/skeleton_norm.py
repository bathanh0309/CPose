import numpy as np

def normalize_skeleton(feat: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton sequence (T, V, C).
    - C=3: (x, y, conf)
    - T: window size
    - V: number of joints (17 for COCO)
    
    Standard normalization:
    1. Subtract center (mid-point of shoulders or hips).
    2. Scale to roughly [-1, 1] based on skeleton height.
    """
    # Create a copy to avoid mutating original
    x = feat.copy()
    
    T, V, C = x.shape
    if V < 13: # Not enough joints for COCO Hip/Shoulder centering
        return x
        
    # Indices for COCO
    # 5, 6: shoulders
    # 11, 12: hips
    
    for t in range(T):
        # Calculate center as midpoint of hips
        hip_center = (x[t, 11, :2] + x[t, 12, :2]) / 2.0
        
        # If hip center is 0 (missing), try shoulders
        if np.all(hip_center == 0):
            hip_center = (x[t, 5, :2] + x[t, 6, :2]) / 2.0
            
        # Subtract center from x, y
        x[t, :, :2] -= hip_center
        
        # Calculate scale: height of person
        # Distance from head (nose: 0) to ankles (15, 16)
        y_coords = x[t, :, 1]
        valid_y = y_coords[feat[t, :, 2] > 0.1]
        
        if len(valid_y) > 0:
            h = np.max(valid_y) - np.min(valid_y)
            if h > 1e-6:
                x[t, :, :2] /= h
    
    return x
