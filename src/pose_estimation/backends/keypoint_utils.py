import numpy as np

def normalize_keypoints(keypoints, center_joint_idx=11):
    """
    Center and scale keypoints for research consistency.
    center_joint_idx: default to hip center in COCO (approx)
    """
    # keypoints: (T, V, C)
    center = keypoints[:, center_joint_idx:center_joint_idx+1, :]
    keypoints = keypoints - center
    
    # Scale by max distance to normalize
    scale = np.max(np.linalg.norm(keypoints, axis=-1))
    if scale > 0:
        keypoints = keypoints / scale
    return keypoints

def get_bone_features(keypoints, skeleton_type='coco'):
    """Calculate bone lengths/angles as secondary features for graph research."""
    # Placeholder for bone connections definition
    pass

def resample_sequence(keypoints, target_len=30):
    """Resample keypoint sequence to fixed length for temporal models."""
    T = keypoints.shape[0]
    if T == target_len:
        return keypoints
    
    indices = np.linspace(0, T - 1, target_len).astype(int)
    return keypoints[indices]
