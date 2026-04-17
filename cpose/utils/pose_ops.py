import numpy as np

def calculate_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle between three points (p2 is vertex)"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_posture_class(keypoints: np.ndarray) -> str:
    """Simple posture classifier based on keypoints heights"""
    # Placeholder: logic for standing/sitting/lying
    return "unknown"
