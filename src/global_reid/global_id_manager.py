import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class GlobalIDManager:
    """
    Global ID Manager implementing TFCS-PAR logic (Time-First Cross-Camera Sequential Pose-ADL-ReID)
    as specified in CPose CLAUDE.md architecture.
    """
    def __init__(self, config: dict):
        self.config = config.get("global_reid", {})
        
        # Thresholds
        self.strong_threshold = self.config.get("strong_threshold", 0.65)
        self.weak_threshold = self.config.get("weak_threshold", 0.45)
        self.confirm_frames = self.config.get("confirm_frames", 3)
        self.max_candidate_age_sec = self.config.get("max_candidate_age_sec", 300)
        
        # Weights
        weights = self.config.get("weights", {})
        self.w_face = weights.get("face", 0.30)
        self.w_body = weights.get("body", 0.20)
        self.w_pose = weights.get("pose", 0.15)
        self.w_height = weights.get("height", 0.10)
        self.w_time = weights.get("time", 0.15)
        self.w_topology = weights.get("topology", 0.10)
        
        # In-memory storage for active global IDs
        self.active_profiles: Dict[str, dict] = {}
        self.next_gid = 1
        
    def _generate_gid(self) -> str:
        gid = f"GID-{self.next_gid:03d}"
        self.next_gid += 1
        return gid
        
    def match(self, camera_id: str, local_track_id: int, features: dict, frame_time: float) -> dict:
        """
        Matches a local track to a Global ID based on TFCS-PAR score.
        features dict should contain:
        - face_embedding: Optional array
        - body_embedding: Optional array
        - pose_signature: Optional array (gait/pose)
        - relative_height: float
        """
        best_match = None
        best_score = 0.0
        
        # 1. Spatiotemporal gating
        valid_candidates = self._get_valid_candidates(camera_id, frame_time)
        
        # 2. Score calculation
        for candidate_gid in valid_candidates:
            score_components = self._compute_scores(self.active_profiles[candidate_gid], features, camera_id, frame_time)
            
            s_total = (
                self.w_face * score_components['face'] +
                self.w_body * score_components['body'] +
                self.w_pose * score_components['pose'] +
                self.w_height * score_components['height'] +
                self.w_time * score_components['time'] +
                self.w_topology * score_components['topology']
            )
            
            if s_total > best_score:
                best_score = s_total
                best_match = candidate_gid
                best_components = score_components
                
        # 3. Decision
        if best_score >= self.strong_threshold:
            assigned_gid = best_match
            state = "ACTIVE"
            match_status = "strong_match"
        elif best_score >= self.weak_threshold:
            assigned_gid = best_match
            state = "PENDING_CONFIRMATION"
            match_status = "weak_match"
        else:
            assigned_gid = self._generate_gid()
            state = "NEW_ID"
            match_status = "no_match"
            best_components = {
                'face': 0.0, 'body': 0.0, 'pose': 0.0, 
                'height': 0.0, 'time': 0.0, 'topology': 0.0
            }
            
        # 4. Update profile
        if state in ["ACTIVE", "NEW_ID"]:
            self._update_profile(assigned_gid, camera_id, features, frame_time)
            
        return {
            "global_id": assigned_gid,
            "state": state,
            "match_status": match_status,
            "score_total": best_score,
            "score_face": best_components.get('face'),
            "score_body": best_components.get('body'),
            "score_pose": best_components.get('pose'),
            "score_height": best_components.get('height'),
            "score_time": best_components.get('time'),
            "score_topology": best_components.get('topology'),
            "topology_allowed": best_components.get('topology') > 0,
            "failure_reason": "OK"
        }
        
    def _get_valid_candidates(self, camera_id: str, frame_time: float) -> List[str]:
        # Implement gating logic based on topology transition windows and max_candidate_age_sec
        candidates = []
        for gid, profile in self.active_profiles.items():
            last_time = profile.get("last_time", 0)
            if (frame_time - last_time) <= self.max_candidate_age_sec:
                # Add topology check here later
                candidates.append(gid)
        return candidates
        
    def _compute_scores(self, profile: dict, features: dict, current_cam: str, current_time: float) -> dict:
        # Placeholder for actual similarity computation
        # Here we will call external modules: fusion_score, topology.py, etc.
        return {
            "face": 0.0,     # Cosine similarity
            "body": 0.0,     # Cosine similarity
            "pose": 0.0,     # Gait/pose signature similarity
            "height": 0.0,   # 1 - min(|h1 - h2| / tau, 1)
            "time": 1.0,     # Gating score
            "topology": 1.0  # Gating score
        }
        
    def _update_profile(self, gid: str, camera_id: str, features: dict, frame_time: float):
        self.active_profiles[gid] = {
            "last_camera": camera_id,
            "last_time": frame_time,
            "features": features
        }
