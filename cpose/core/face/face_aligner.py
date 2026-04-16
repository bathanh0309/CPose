from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import cv2

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class FaceCrop:
    image: np.ndarray    # HxWxC, RGB/BGR
    bbox: BBox
    cam_id: str
    track_id: int        # tracker ID (single/multi-cam)
    timestamp: float

class FaceAligner:
    """
    - Takes frame + bbox (person or face) -> crops & aligns face.
    - Focuses on geometry and cropping logic.
    """

    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size

    def crop_from_face_bbox(
        self,
        frame: np.ndarray,
        face_bbox: BBox,
        cam_id: str,
        track_id: int,
        timestamp: float,
    ) -> FaceCrop:
        """
        Crop directly from a face bounding box (requires face detector).
        """
        h, w = frame.shape[:2]
        x1 = max(0, face_bbox.x1)
        y1 = max(0, face_bbox.y1)
        x2 = min(w, face_bbox.x2)
        y2 = min(h, face_bbox.y2)

        face = frame[y1:y2, x1:x2].copy()
        face = self._resize(face)
        return FaceCrop(face, BBox(x1, y1, x2, y2), cam_id, track_id, timestamp)

    def crop_from_person_bbox(
        self,
        frame: np.ndarray,
        person_bbox: BBox,
        cam_id: str,
        track_id: int,
        timestamp: float,
    ) -> FaceCrop:
        """
        Heuristic fallback: crop the upper portion of the person bbox as the face area.
        Assumes face is roughly in the top 40% of the person bbox.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox.x1, person_bbox.y1, person_bbox.x2, person_bbox.y2

        # Heuristic: top 40%
        face_y2 = y1 + int((y2 - y1) * 0.4)
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, face_y2 = min(w, x2), min(h, face_y2)

        face = frame[y1:face_y2, x1:x2].copy()
        if face.size == 0:
            # Return empty or placeholder if crop fails
            face = np.zeros((*self.target_size, 3), dtype=np.uint8)
        else:
            face = self._resize(face)
            
        return FaceCrop(face, BBox(x1, y1, x2, face_y2), cam_id, track_id, timestamp)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.target_size)
