from app.core.tracking_types import Track

def track_to_json(trk: Track) -> dict:
    """
    Standard serialization of a Track object to JSON-compatible dictionary.
    Includes Pose, Tracking, ADL, ReID, and FaceID information.
    """
    return {
        "cam_id": trk.cam_id,
        "local_id": trk.local_id,
        "global_id": trk.global_id,
        "bbox": (
            {
                "x1": trk.bbox.x1,
                "y1": trk.bbox.y1,
                "x2": trk.bbox.x2,
                "y2": trk.bbox.y2,
            } if trk.bbox else None
        ),
        "bbox_score": trk.bbox_score,
        "track_confidence": trk.track_confidence,

        "action": {
            "label": trk.action.label,
            "score": trk.action.score,
        } if trk.action and trk.action.label else None,

        "identity": {
            "global_id": trk.global_id,
            "identity_id": trk.identity_id,
            "name": trk.identity_name,
            "source": trk.meta.get("identity_source", "unknown"),
            "face_similarity": trk.face_similarity,
            "reid_score": trk.reid_score,
        },

        "pose": None if trk.pose is None else {
            "num_joints": int(trk.pose.keypoints.shape[0]),
            # Optionally filter keypoints to reduce payload size if needed
            "keypoints": trk.pose.keypoints.tolist(),
        },

        "state": {
            "is_active": trk.is_active,
            "is_occluded": trk.is_occluded,
            "lost_frames": trk.lost_frames,
        },
    }
