from .base import BasePoseEstimator
from .yolo_pose import YOLOPoseEstimator

def build_pose_estimator(name: str, cfg: dict) -> BasePoseEstimator:
    """
    Factory to create pose estimators.
    Supported: 'yolo_pose', 'rtmpose' (placeholder)
    """
    if name.lower() in ['yolo_pose', 'yolo']:
        return YOLOPoseEstimator(
            model_path=cfg.get('model_path'),
            conf_threshold=cfg.get('conf', 0.25)
        )
    elif name.lower() == 'rtmpose':
        # Placeholder for RTMPose
        from .rtmpose import RTMPoseWrapper
        return RTMPoseWrapper(model_path=cfg.get('model_path'))
    else:
        raise ValueError(f"Unknown pose estimator type: {name}")
