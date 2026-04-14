"""
GlobalIDManager - Master-Slave Multi-Camera ID Assignment
CRITICAL: Cam1 is the REGISTRATION camera - creates GlobalID + Profile.
Cam2/3/4 are SLAVES - match only.

Author: Senior MLOps Engineer
Date: 2026-02-02
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import threading

from app.storage.persistence import PersistenceManager
from app.storage.vector_db import HybridMatcher, VectorDatabase
from app.utils.runtime_config import get_runtime_section

logger = logging.getLogger(__name__)

_GLOBAL_ID_DEFAULTS = get_runtime_section("global_id")
QUALITY_UPDATE_THRESHOLD = float(_GLOBAL_ID_DEFAULTS.get("quality_update_threshold", 0.7))

_DEFAULT_TRANSITION_WINDOWS = {
    ("cam01", "cam02"): (0.0, 60.0),
    ("cam02", "cam02"): (0.0, 120.0),
    ("cam02", "cam03"): (0.0, 60.0),
    ("cam03", "cam02"): (10.0, 120.0),
    ("cam03", "cam03"): (0.0, 180.0),
    ("cam03", "cam04"): (20.0, 180.0),
    ("cam04", "cam03"): (20.0, 180.0),
    ("cam04", "cam04"): (0.0, 300.0),
}


def _normalize_camera_id(camera: str | None) -> str:
    if not camera:
        return ""

    camera = str(camera).strip().lower()
    if not camera.startswith("cam"):
        return camera

    digits = "".join(ch for ch in camera if ch.isdigit())
    if not digits:
        return camera
    return f"cam{int(digits):02d}"


def _parse_transition_windows(raw_windows) -> Dict[Tuple[str, str], Tuple[float, float]]:
    windows = dict(_DEFAULT_TRANSITION_WINDOWS)
    if not isinstance(raw_windows, dict):
        return windows

    for raw_key, raw_value in raw_windows.items():
        source = None
        target = None

        if isinstance(raw_key, str) and "->" in raw_key:
            left, right = raw_key.split("->", 1)
            source = _normalize_camera_id(left)
            target = _normalize_camera_id(right)
        elif isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
            source = _normalize_camera_id(raw_key[0])
            target = _normalize_camera_id(raw_key[1])

        if not source or not target:
            continue
        if not isinstance(raw_value, (tuple, list)) or len(raw_value) != 2:
            continue

        try:
            min_s = float(raw_value[0])
            max_s = float(raw_value[1])
        except (TypeError, ValueError):
            continue

        if min_s > max_s:
            min_s, max_s = max_s, min_s
        windows[(source, target)] = (min_s, max_s)

    return windows


class GlobalIDManager:
    """
    Centralized GlobalID management with strict Master-Slave architecture.
    
    Rules:
    1. MASTER camera (cam1) is the REGISTRATION source - creates GlobalIDs + profiles
    2. SLAVE cameras (cam2, cam3, cam4) can ONLY match existing GlobalIDs or assign UNK
    3. GlobalIDs are sequential integers: G1, G2, G3, ...
    4. No ID recycling - once assigned, never reused
    5. State persists across restarts
    """
    
    def __init__(
        self,
        master_camera: str,
        slave_cameras: List[str],
        persistence: PersistenceManager,
        vector_db: VectorDatabase,
        config: dict
    ):
        """
        Args:
            master_camera: Camera ID that creates GlobalIDs (e.g., 'cam2')
            slave_cameras: List of slave camera IDs (e.g., ['cam3', 'cam4'])
            persistence: Persistence layer
            vector_db: Vector database for ANN search
            config: Configuration dict
        """
        self.master_camera = master_camera
        self.slave_cameras = slave_cameras
        self.persistence = persistence
        self.vector_db = vector_db
        self.matcher = HybridMatcher(vector_db)
        config = config or {}
        
        # Configuration
        self.strong_threshold = config.get("strong_threshold", _GLOBAL_ID_DEFAULTS.get("strong_threshold", 0.65))
        self.weak_threshold = config.get("weak_threshold", _GLOBAL_ID_DEFAULTS.get("weak_threshold", 0.45))
        self.confirm_frames = config.get("confirm_frames", _GLOBAL_ID_DEFAULTS.get("confirm_frames", 3))
        self.top_k_candidates = config.get("top_k_candidates", _GLOBAL_ID_DEFAULTS.get("top_k_candidates", 20))
        self.use_hungarian = config.get("use_hungarian", _GLOBAL_ID_DEFAULTS.get("use_hungarian", True))
        self.transition_windows = _parse_transition_windows(
            config.get("transition_windows", _GLOBAL_ID_DEFAULTS.get("transition_windows", {}))
        )
        
        # UNK handling
        self.unk_namespace = config.get("unk_namespace", "global")
        self.max_unk_per_video = config.get(
            "max_unk_per_video",
            _GLOBAL_ID_DEFAULTS.get("max_unk_per_video", 10),
        )
        
        # Temporal voting: track_id -> {global_id: count}
        self.match_history = defaultdict(lambda: defaultdict(int))
        
        # UNK tracking
        if self.unk_namespace == 'global':
            self.next_unk_id = 1
        else:
            self.next_unk_id = {cam: 1 for cam in slave_cameras}
        
        # Spatial persistence for UNK resurrection
        self.last_unk_bboxes = {}  # camera -> {unk_id: bbox}
        self.iou_threshold = config.get(
            "iou_resurrection_threshold",
            _GLOBAL_ID_DEFAULTS.get("iou_resurrection_threshold", 0.3),
        )
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing state from persistence
        self._load_state()
        
        logger.info(
            f"GlobalIDManager initialized: master={master_camera}, "
            f"slaves={slave_cameras}, thresholds=[{self.weak_threshold}, {self.strong_threshold}]"
        )
    
    def _load_state(self):
        """Load state from persistence and rebuild vector index."""
        stats = self.persistence.get_statistics()
        logger.info(f"Loading state: {stats}")
        
        # Rebuild vector index
        embeddings, global_ids = self.persistence.get_all_embeddings(active_only=True)
        if len(embeddings) > 0:
            self.vector_db.rebuild(embeddings, global_ids)
            logger.info(f"Rebuilt vector index with {len(global_ids)} GlobalIDs")
    
    def assign_id(
        self,
        camera: str,
        track_id: int,
        embedding: np.ndarray,
        bbox: List[int],
        frame_time: float,
        quality_score: float = 1.0
    ) -> Tuple[str, dict]:
        """
        Main entry point for ID assignment.
        
        Args:
            camera: Camera ID
            track_id: Local track ID from tracker
            embedding: Feature embedding (L2-normalized)
            bbox: Bounding box [x1, y1, x2, y2]
            frame_time: Timestamp of current frame
            quality_score: Crop quality [0, 1] (for feature update gating)
        
        Returns:
            assigned_id: String like 'G42' or 'UNK3'
            metadata: Dict with match info (score, state, etc.)
        """
        with self.lock:
            if camera == self.master_camera:
                return self._assign_master(
                    track_id, embedding, bbox, frame_time, quality_score
                )
            elif camera in self.slave_cameras:
                return self._assign_slave(
                    camera, track_id, embedding, bbox, frame_time, quality_score
                )
            else:
                logger.warning(f"Unknown camera: {camera}")
                return "UNK_UNKNOWN", {'state': 'ERROR'}
    
    def _assign_master(
        self,
        track_id: int,
        embedding: np.ndarray,
        bbox: List[int],
        frame_time: float,
        quality_score: float
    ) -> Tuple[str, dict]:
        """
        Master camera assignment logic.
        
        Strategy:
        1. Try to match against existing GlobalIDs
        2. If no strong match, CREATE new GlobalID
        3. Update feature bank for matched/created ID
        """
        # Search for match
        candidates = self.matcher.match(
            embedding,
            allowed_ids=None,  # Master can match anyone
            top_k=self.top_k_candidates
        )
        
        metadata = {'state': 'PENDING', 'candidates': []}
        
        if len(candidates) > 0:
            best_match = candidates[0]
            score = best_match['score']
            global_id = best_match['global_id']
            
            metadata['candidates'] = candidates[:5]  # Top 5 for logging
            metadata['best_score'] = score
            
            # Strong match
            if score >= self.strong_threshold:
                # Temporal voting for stability
                self.match_history[track_id][global_id] += 1
                
                if self.match_history[track_id][global_id] >= self.confirm_frames:
                    # CONFIRMED
                    metadata['state'] = 'CONFIRMED'
                    metadata['votes'] = self.match_history[track_id][global_id]
                    
                    # Update feature bank
                    if quality_score >= QUALITY_UPDATE_THRESHOLD:
                        appearance_time = self._coerce_frame_time(frame_time)
                        self.persistence.update_appearance(
                            global_id, 
                            self.master_camera,
                            embedding,
                            bbox,
                            timestamp=appearance_time,
                        )
                        # Update vector index
                        self._update_vector_index(global_id, embedding)
                    
                    # Clear voting history for this track
                    self.match_history[track_id].clear()
                    
                    return f"G{global_id}", metadata
                else:
                    # Still pending confirmation
                    metadata['state'] = 'PENDING'
                    metadata['votes'] = self.match_history[track_id][global_id]
                    return f"G{global_id}*", metadata  # * indicates pending
            
            # Weak match or no match - create new ID
        
        # CREATE NEW GLOBALID (Master privilege)
        new_id = self.persistence.get_next_global_id()
        
        # Register
        self.persistence.register_global_id(
            new_id,
            self.master_camera,
            embedding,
            bbox,
            timestamp=self._coerce_frame_time(frame_time),
        )
        
        # Add to vector index
        self.vector_db.add(
            embedding.reshape(1, -1),
            np.array([new_id])
        )
        
        metadata['state'] = 'NEW_ID'
        metadata['global_id'] = new_id
        
        logger.info(f"Master created new GlobalID: {new_id}")
        
        return f"G{new_id}", metadata
    
    def _assign_slave(
        self,
        camera: str,
        track_id: int,
        embedding: np.ndarray,
        bbox: List[int],
        frame_time: float,
        quality_score: float
    ) -> Tuple[str, dict]:
        """
        Slave camera assignment logic.
        
        Strategy:
        1. ONLY match against existing GlobalIDs from master
        2. Apply spatiotemporal gating (camera transition rules)
        3. If no match, assign UNK (never create GlobalID)
        4. Resurrect UNK via IoU if detection flickers
        """
        # Get valid candidate GlobalIDs (with spatiotemporal filtering)
        valid_candidates = self._get_valid_candidates(camera, frame_time)
        
        if len(valid_candidates) == 0:
            # No valid candidates - must be UNK or GHOST
            return self._assign_unk(camera, track_id, bbox, 'NO_CANDIDATES')
        
        # Search for match
        candidates = self.matcher.match(
            embedding,
            allowed_ids=valid_candidates,
            top_k=min(self.top_k_candidates, len(valid_candidates))
        )
        
        metadata = {'state': 'PENDING', 'candidates': []}
        
        if len(candidates) == 0:
            return self._assign_unk(camera, track_id, bbox, 'NO_MATCH')
        
        best_match = candidates[0]
        score = best_match['score']
        global_id = best_match['global_id']
        
        metadata['candidates'] = candidates[:5]
        metadata['best_score'] = score
        
        # Strong match
        if score >= self.strong_threshold:
            # Temporal voting
            self.match_history[track_id][global_id] += 1
            
            if self.match_history[track_id][global_id] >= self.confirm_frames:
                # CONFIRMED
                metadata['state'] = 'CONFIRMED'
                metadata['votes'] = self.match_history[track_id][global_id]
                
                # Update appearance (but NOT create new ID)
                if quality_score >= QUALITY_UPDATE_THRESHOLD:
                    appearance_time = self._coerce_frame_time(frame_time)
                    self.persistence.update_appearance(
                        global_id,
                        camera,
                        embedding,
                        bbox,
                        timestamp=appearance_time,
                    )
                    # Update vector index
                    self._update_vector_index(global_id, embedding)
                
                # Clear voting
                self.match_history[track_id].clear()
                
                return f"G{global_id}", metadata
            else:
                # Pending
                metadata['state'] = 'PENDING'
                metadata['votes'] = self.match_history[track_id][global_id]
                return f"G{global_id}*", metadata
        
        elif score >= self.weak_threshold:
            # Weak match - keep pending
            metadata['state'] = 'WEAK'
            return self._assign_unk(camera, track_id, bbox, 'WEAK_MATCH')
        
        else:
            # Below threshold
            return self._assign_unk(camera, track_id, bbox, 'BELOW_THRESHOLD')
    
    def _assign_unk(
        self,
        camera: str,
        track_id: int,
        bbox: List[int],
        reason: str
    ) -> Tuple[str, dict]:
        """
        Assign UNK ID with spatial persistence (IoU resurrection).
        
        Args:
            camera: Camera ID
            track_id: Track ID
            bbox: Bounding box
            reason: Why assigned UNK
        
        Returns:
            unk_id: Like 'UNK3'
            metadata: Dict with state
        """
        # Try to resurrect old UNK via IoU
        if camera in self.last_unk_bboxes:
            for unk_id, old_bbox in self.last_unk_bboxes[camera].items():
                iou = self._compute_iou(bbox, old_bbox)
                if iou > self.iou_threshold:
                    # Resurrect
                    logger.debug(f"Resurrected {unk_id} via IoU={iou:.2f}")
                    self.last_unk_bboxes[camera][unk_id] = bbox
                    return unk_id, {
                        'state': 'UNK_RESURRECTED',
                        'iou': iou,
                        'reason': reason
                    }
        
        # Create new UNK
        if self.unk_namespace == 'global':
            unk_id = f"UNK{self.next_unk_id}"
            self.next_unk_id += 1
        else:
            unk_id = f"UNK{self.next_unk_id[camera]}"
            self.next_unk_id[camera] += 1
        
        # Store bbox for future resurrection
        if camera not in self.last_unk_bboxes:
            self.last_unk_bboxes[camera] = {}
        self.last_unk_bboxes[camera][unk_id] = bbox
        
        # Limit UNK count
        if len(self.last_unk_bboxes[camera]) > self.max_unk_per_video:
            # Remove oldest
            oldest = list(self.last_unk_bboxes[camera].keys())[0]
            del self.last_unk_bboxes[camera][oldest]
        
        return unk_id, {
            'state': 'UNK_NEW',
            'reason': reason
        }
    
    def _get_valid_candidates(self, camera: str, frame_time: float) -> List[int]:
        """
        Get list of valid GlobalID candidates for slave camera.
        Apply spatiotemporal filtering.
        """
        target_camera = _normalize_camera_id(camera)
        current_dt = self._coerce_frame_time(frame_time)
        if current_dt is None:
            logger.debug("Invalid frame_time=%s. Rejecting candidates for %s.", frame_time, target_camera)
            return []

        candidate_rows = self.persistence.get_active_global_id_states()
        valid_ids: List[int] = []

        for row in candidate_rows:
            global_id = int(row["global_id"])
            last_camera = _normalize_camera_id(row.get("last_camera"))
            last_seen_at = row.get("last_seen_at")

            if not last_camera or not last_seen_at:
                continue

            previous_dt = self._parse_iso_datetime(last_seen_at)
            if previous_dt is None:
                continue

            window = self.transition_windows.get((last_camera, target_camera))
            if window is None:
                continue

            delta_s = (current_dt - previous_dt).total_seconds()
            min_s, max_s = window
            if min_s <= delta_s <= max_s:
                valid_ids.append(global_id)

        logger.debug(
            "Spatiotemporal gating for %s at %s kept %d/%d candidates",
            target_camera,
            current_dt.isoformat(),
            len(valid_ids),
            len(candidate_rows),
        )
        return valid_ids

    @staticmethod
    def _coerce_frame_time(frame_time) -> Optional[datetime]:
        if isinstance(frame_time, datetime):
            return frame_time
        if isinstance(frame_time, str):
            try:
                return datetime.fromisoformat(frame_time)
            except ValueError:
                return None
        try:
            return datetime.fromtimestamp(float(frame_time))
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    @staticmethod
    def _parse_iso_datetime(raw_value: str | None) -> Optional[datetime]:
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(raw_value)
        except ValueError:
            return None
    
    def _update_vector_index(self, global_id: int, new_embedding: np.ndarray):
        """
        Update vector index with new embedding (EMA style).
        The current VectorDatabase does not expose a safe in-place update API,
        so we rebuild from persistence to keep the search index consistent.
        """
        embeddings, global_ids = self.persistence.get_all_embeddings(active_only=True)
        if len(embeddings) == 0:
            return

        self.vector_db.rebuild(embeddings, global_ids)
        logger.debug("Vector index refreshed after GlobalID %s update", global_id)
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def get_statistics(self) -> dict:
        """Get statistics for monitoring."""
        persist_stats = self.persistence.get_statistics()
        vector_stats = self.vector_db.get_stats()
        
        return {
            'persistence': persist_stats,
            'vector_db': vector_stats,
            'unk_count': self.next_unk_id if self.unk_namespace == 'global' else sum(self.next_unk_id.values()),
            'active_votes': len(self.match_history)
        }
