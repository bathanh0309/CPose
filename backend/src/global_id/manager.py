"""
Global ID Manager - Cam2 Master Edition

Key Features:
1. ONLY Camera 2 creates new Global IDs (sequential: 1, 2, 3...)
2. Camera 3/4 can only MATCH existing IDs or return UNKNOWN
3. Anti-flicker: stabilization + cooldown for re-attachment
4. Open-set: Two-threshold association (strong/weak/pending)
5. Spatiotemporal gating using camera graph
6. Gallery update (EMA) for domain shift adaptation
7. Persistent storage (SQLite)
"""

import numpy as np
import time
import sqlite3
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a local track in a camera"""
    track_id: int
    camera_id: str
    first_seen: float
    last_seen: float
    embeddings: List[np.ndarray] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    frame_count: int = 0
    is_stable: bool = False
    quality_frames: int = 0


@dataclass
class GlobalIdentity:
    """Represents a global identity across cameras"""
    global_id: int
    gallery_embedding: np.ndarray
    camera_history: Dict[str, float] = field(default_factory=dict)  # camera_id -> last_seen_time
    creation_time: float = 0.0
    creation_camera: str = ""
    total_appearances: int = 0
    update_count: int = 0


@dataclass
class PendingMatch:
    """Pending match decision"""
    track_id: int
    camera_id: str
    candidate_global_id: Optional[int]
    score: float
    start_time: float
    frame_count: int
    embeddings: List[np.ndarray] = field(default_factory=list)


class CameraGraph:
    """Camera graph for spatiotemporal gating"""
    
    def __init__(self, edges: List[dict]):
        self.edges = {}
        for edge in edges:
            key = (edge['from'], edge['to'])
            self.edges[key] = {
                'min_time': edge['min_time'],
                'max_time': edge['max_time']
            }
    
    def is_valid_transition(
        self,
        from_camera: str,
        to_camera: str,
        time_diff: float
    ) -> bool:
        """Check if transition time is valid"""
        key = (from_camera, to_camera)
        if key not in self.edges:
            # No constraint defined, allow
            return True
        
        edge = self.edges[key]
        return edge['min_time'] <= time_diff <= edge['max_time']


class GlobalIDManager:
    """
    Global ID Manager with Cam2 as Master
    
    Architecture:
    - Master camera (cam2): Creates sequential Global IDs (1, 2, 3...)
    - Slave cameras (cam3, cam4): Only match or return UNKNOWN
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Master camera
        self.master_camera = config.get('master_camera', 'cam2')
        
        # Thresholds
        self.strong_threshold = config.get('global_id', {}).get('strong_threshold', 0.65)
        self.weak_threshold = config.get('global_id', {}).get('weak_threshold', 0.45)
        
        # Anti-flicker
        self.min_frames_stable = config.get('global_id', {}).get('min_frames_stable', 15)
        self.min_quality_frames = config.get('global_id', {}).get('min_quality_frames', 10)
        self.cooldown_seconds = config.get('global_id', {}).get('cooldown_seconds', 10)
        self.max_cooldown_tracks = config.get('global_id', {}).get('max_cooldown_tracks', 50)
        
        # Gallery update
        self.enable_gallery_update = config.get('global_id', {}).get('enable_gallery_update', True)
        self.gallery_update_alpha = config.get('global_id', {}).get('gallery_update_alpha', 0.3)
        self.update_threshold = config.get('global_id', {}).get('update_threshold', 0.70)
        
        # Pending state
        self.max_pending_frames = config.get('global_id', {}).get('max_pending_frames', 30)
        self.pending_decision_frames = config.get('global_id', {}).get('pending_decision_frames', 10)
        
        # Camera graph
        camera_graph_edges = config.get('camera_graph', {}).get('edges', [])
        self.camera_graph = CameraGraph(camera_graph_edges)
        
        # Storage
        self.storage_path = config.get('global_id', {}).get('storage_path', 'backend/data/global_id_state.db')
        
        # State
        self.next_global_id = 1
        self.global_identities: Dict[int, GlobalIdentity] = {}
        
        # Active tracks per camera
        self.active_tracks: Dict[str, Dict[int, Track]] = defaultdict(dict)
        
        # Cooldown tracks (recently lost, for re-attachment)
        self.cooldown_tracks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_cooldown_tracks))
        
        # Pending matches
        self.pending_matches: Dict[Tuple[str, int], PendingMatch] = {}
        
        # Unknown counter per camera
        self.unknown_counters: Dict[str, int] = defaultdict(int)
        
        # Track to global ID mapping
        self.track_to_global: Dict[Tuple[str, int], int] = {}
        
        # Load state if exists
        self._load_state()
        
        logger.info(f"GlobalIDManager initialized. Master: {self.master_camera}, Next ID: {self.next_global_id}")
    
    def update_track(
        self,
        camera_id: str,
        track_id: int,
        embedding: np.ndarray,
        bbox: List[float],
        confidence: float,
        timestamp: float
    ) -> Tuple[Optional[int], str, float]:
        """
        Update track and get global ID assignment
        
        Returns:
            (global_id, status, confidence)
            - global_id: int or None (for UNKNOWN)
            - status: 'known', 'unknown', 'pending', 'new'
            - confidence: match confidence (0-1)
        """
        # Get or create track
        if track_id not in self.active_tracks[camera_id]:
            self.active_tracks[camera_id][track_id] = Track(
                track_id=track_id,
                camera_id=camera_id,
                first_seen=timestamp,
                last_seen=timestamp
            )
        
        track = self.active_tracks[camera_id][track_id]
        
        # Update track
        track.last_seen = timestamp
        track.embeddings.append(embedding)
        track.bboxes.append(bbox)
        track.confidences.append(confidence)
        track.frame_count += 1
        
        # Check quality
        if self._is_high_quality(bbox, confidence):
            track.quality_frames += 1
        
        # Check stability
        if track.frame_count >= self.min_frames_stable and track.quality_frames >= self.min_quality_frames:
            track.is_stable = True
        
        # Check if already assigned
        key = (camera_id, track_id)
        if key in self.track_to_global:
            global_id = self.track_to_global[key]
            # Update last seen in global identity
            if global_id in self.global_identities:
                self.global_identities[global_id].camera_history[camera_id] = timestamp
                
                # Gallery update if confident match
                if self.enable_gallery_update:
                    self._maybe_update_gallery(global_id, embedding, 1.0)  # Existing assignment = 1.0
            
            return global_id, 'known', 1.0
        
        # Try to assign global ID
        return self._assign_global_id(track, timestamp)
    
    def _assign_global_id(
        self,
        track: Track,
        timestamp: float
    ) -> Tuple[Optional[int], str, float]:
        """
        Assign global ID to a track
        
        Logic:
        - Master camera (cam2): Can create new IDs after stabilization
        - Slave cameras (cam3/cam4): Only match existing IDs or UNKNOWN
        """
        # Get average embedding
        avg_embedding = np.mean(track.embeddings[-10:], axis=0)  # Use last 10 embeddings
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        # Step 1: Try cooldown re-attachment (anti-flicker)
        cooldown_result = self._try_cooldown_reattach(track, avg_embedding, timestamp)
        if cooldown_result is not None:
            global_id, score = cooldown_result
            self.track_to_global[(track.camera_id, track.track_id)] = global_id
            logger.info(f"[{track.camera_id}] T{track.track_id} re-attached to G{global_id} (cooldown, score={score:.3f})")
            return global_id, 'known', score
        
        # Step 2: Try gallery matching
        best_match_id, best_score = self._match_gallery(avg_embedding, track.camera_id, timestamp)
        
        # Step 3: Decision based on threshold
        if best_score >= self.strong_threshold:
            # Strong match
            if self._validate_spatiotemporal(best_match_id, track.camera_id, timestamp):
                self.track_to_global[(track.camera_id, track.track_id)] = best_match_id
                self.global_identities[best_match_id].camera_history[track.camera_id] = timestamp
                self.global_identities[best_match_id].total_appearances += 1
                
                # Gallery update
                if self.enable_gallery_update and best_score >= self.update_threshold:
                    self._maybe_update_gallery(best_match_id, avg_embedding, best_score)
                
                logger.info(f"[{track.camera_id}] T{track.track_id} matched to G{best_match_id} (score={best_score:.3f})")
                return best_match_id, 'known', best_score
            else:
                logger.warning(f"[{track.camera_id}] T{track.track_id} failed spatiotemporal check for G{best_match_id}")
                best_match_id = None
                best_score = 0.0
        
        elif best_score > self.weak_threshold:
            # Pending - need more frames
            key = (track.camera_id, track.track_id)
            if key not in self.pending_matches:
                self.pending_matches[key] = PendingMatch(
                    track_id=track.track_id,
                    camera_id=track.camera_id,
                    candidate_global_id=best_match_id,
                    score=best_score,
                    start_time=timestamp,
                    frame_count=1,
                    embeddings=[avg_embedding]
                )
                logger.debug(f"[{track.camera_id}] T{track.track_id} pending for G{best_match_id} (score={best_score:.3f})")
            else:
                pending = self.pending_matches[key]
                pending.embeddings.append(avg_embedding)
                pending.frame_count += 1
                
                # Decision time
                if pending.frame_count >= self.pending_decision_frames:
                    # Re-evaluate with more embeddings
                    combined_emb = np.mean(pending.embeddings, axis=0)
                    combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)
                    
                    new_id, new_score = self._match_gallery(combined_emb, track.camera_id, timestamp)
                    
                    if new_score >= self.strong_threshold and self._validate_spatiotemporal(new_id, track.camera_id, timestamp):
                        # Upgrade to known
                        self.track_to_global[key] = new_id
                        self.global_identities[new_id].camera_history[track.camera_id] = timestamp
                        del self.pending_matches[key]
                        logger.info(f"[{track.camera_id}] T{track.track_id} upgraded to G{new_id} (score={new_score:.3f})")
                        return new_id, 'known', new_score
                    else:
                        # Downgrade to unknown
                        del self.pending_matches[key]
                        return self._handle_unknown(track)
                
                elif (timestamp - pending.start_time) * 1000 > self.max_pending_frames * (1000 / 30):
                    # Timeout
                    del self.pending_matches[key]
                    return self._handle_unknown(track)
            
            return None, 'pending', best_score
        
        # Step 4: Weak or no match
        else:
            # Master camera: create new ID if stable
            if track.camera_id == self.master_camera:
                if track.is_stable:
                    return self._create_new_global_id(track, avg_embedding, timestamp)
                else:
                    # Not stable yet
                    return None, 'pending', 0.0
            
            # Slave cameras: UNKNOWN
            else:
                return self._handle_unknown(track)
    
    def _create_new_global_id(
        self,
        track: Track,
        embedding: np.ndarray,
        timestamp: float
    ) -> Tuple[int, str, float]:
        """Create new global ID (only for master camera)"""
        if track.camera_id != self.master_camera:
            raise ValueError(f"Only master camera ({self.master_camera}) can create new IDs!")
        
        global_id = self.next_global_id
        self.next_global_id += 1
        
        # Create global identity
        identity = GlobalIdentity(
            global_id=global_id,
            gallery_embedding=embedding.copy(),
            camera_history={track.camera_id: timestamp},
            creation_time=timestamp,
            creation_camera=track.camera_id,
            total_appearances=1
        )
        
        self.global_identities[global_id] = identity
        self.track_to_global[(track.camera_id, track.track_id)] = global_id
        
        logger.info(f"[{track.camera_id}] T{track.track_id} created new G{global_id}")
        
        return global_id, 'new', 1.0
    
    def _handle_unknown(self, track: Track) -> Tuple[None, str, float]:
        """Handle unknown person (slave cameras only)"""
        # Assign UNKNOWN label (not a global ID)
        # We don't store this in track_to_global to keep generating UNK labels
        logger.debug(f"[{track.camera_id}] T{track.track_id} marked as UNKNOWN")
        return None, 'unknown', 0.0
    
    def _match_gallery(
        self,
        embedding: np.ndarray,
        camera_id: str,
        timestamp: float
    ) -> Tuple[Optional[int], float]:
        """Match embedding against gallery"""
        if not self.global_identities:
            return None, 0.0
        
        best_id = None
        best_score = 0.0
        
        for global_id, identity in self.global_identities.items():
            # Compute cosine similarity
            score = np.dot(embedding, identity.gallery_embedding)
            
            if score > best_score:
                best_score = score
                best_id = global_id
        
        return best_id, best_score
    
    def _validate_spatiotemporal(
        self,
        global_id: int,
        target_camera: str,
        timestamp: float
    ) -> bool:
        """Validate spatiotemporal constraints"""
        if global_id not in self.global_identities:
            return False
        
        identity = self.global_identities[global_id]
        
        # Check each previous camera appearance
        for prev_camera, prev_time in identity.camera_history.items():
            if prev_camera == target_camera:
                continue
            
            time_diff = timestamp - prev_time
            
            if not self.camera_graph.is_valid_transition(prev_camera, target_camera, time_diff):
                logger.debug(
                    f"Spatiotemporal violation: G{global_id} {prev_camera}->{target_camera} "
                    f"time={time_diff:.1f}s"
                )
                return False
        
        return True
    
    def _try_cooldown_reattach(
        self,
        track: Track,
        embedding: np.ndarray,
        timestamp: float
    ) -> Optional[Tuple[int, float]]:
        """Try to re-attach to recently lost tracks (anti-flicker)"""
        cooldown_list = self.cooldown_tracks[track.camera_id]
        
        best_match = None
        best_score = 0.0
        
        for old_track in cooldown_list:
            # Check time constraint
            time_diff = timestamp - old_track.last_seen
            if time_diff > self.cooldown_seconds:
                continue
            
            # Check if old track had global ID
            old_key = (old_track.camera_id, old_track.track_id)
            if old_key not in self.track_to_global:
                continue
            
            # Compute IoU with last bbox
            if track.bboxes and old_track.bboxes:
                iou = self._compute_iou(track.bboxes[-1], old_track.bboxes[-1])
            else:
                iou = 0.0
            
            # Compute embedding similarity
            if old_track.embeddings:
                old_emb = np.mean(old_track.embeddings[-5:], axis=0)
                old_emb = old_emb / (np.linalg.norm(old_emb) + 1e-8)
                emb_sim = np.dot(embedding, old_emb)
            else:
                emb_sim = 0.0
            
            # Combined score
            score = 0.7 * emb_sim + 0.3 * iou
            
            if score > best_score and score > self.strong_threshold:
                best_score = score
                best_match = self.track_to_global[old_key]
        
        if best_match is not None:
            return best_match, best_score
        
        return None
    
    def _maybe_update_gallery(self, global_id: int, embedding: np.ndarray, score: float):
        """Update gallery embedding using EMA"""
        if score < self.update_threshold:
            return
        
        identity = self.global_identities[global_id]
        
        # EMA update
        old_emb = identity.gallery_embedding
        new_emb = self.gallery_update_alpha * embedding + (1 - self.gallery_update_alpha) * old_emb
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        
        identity.gallery_embedding = new_emb
        identity.update_count += 1
        
        logger.debug(f"Gallery updated for G{global_id} (count={identity.update_count})")
    
    def _is_high_quality(self, bbox: List[float], confidence: float) -> bool:
        """Check if detection is high quality"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        min_height = self.config.get('reid', {}).get('min_bbox_height', 40)
        min_width = self.config.get('reid', {}).get('min_bbox_width', 20)
        min_conf = self.config.get('detection', {}).get('conf_threshold', 0.3)
        
        return height >= min_height and width >= min_width and confidence >= min_conf
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def remove_track(self, camera_id: str, track_id: int):
        """Remove track and add to cooldown"""
        if track_id in self.active_tracks[camera_id]:
            track = self.active_tracks[camera_id][track_id]
            
            # Add to cooldown if has global ID
            key = (camera_id, track_id)
            if key in self.track_to_global:
                self.cooldown_tracks[camera_id].append(track)
                logger.debug(f"[{camera_id}] T{track_id} moved to cooldown")
            
            del self.active_tracks[camera_id][track_id]
    
    def get_global_label(self, camera_id: str, track_id: int) -> str:
        """Get display label for a track"""
        key = (camera_id, track_id)
        
        if key in self.track_to_global:
            global_id = self.track_to_global[key]
            return f"G{global_id}"
        
        if key in self.pending_matches:
            return "PENDING"
        
        # Unknown
        return "UNKNOWN"
    
    def _save_state(self):
        """Save state to SQLite"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    global_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    creation_time REAL,
                    creation_camera TEXT,
                    total_appearances INTEGER,
                    update_count INTEGER
                )
            ''')
            
            # Save metadata
            cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('next_global_id', self.next_global_id))
            
            # Save identities
            cursor.execute('DELETE FROM identities')
            for global_id, identity in self.global_identities.items():
                cursor.execute(
                    'INSERT INTO identities VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        global_id,
                        pickle.dumps(identity.gallery_embedding),
                        identity.creation_time,
                        identity.creation_camera,
                        identity.total_appearances,
                        identity.update_count
                    )
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"State saved: {len(self.global_identities)} identities, next_id={self.next_global_id}")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load state from SQLite"""
        if not Path(self.storage_path).exists():
            logger.info("No saved state found. Starting fresh.")
            return
        
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Load metadata
            cursor.execute('SELECT value FROM metadata WHERE key = ?', ('next_global_id',))
            row = cursor.fetchone()
            if row:
                self.next_global_id = row[0]
            
            # Load identities
            cursor.execute('SELECT * FROM identities')
            for row in cursor.fetchall():
                global_id, embedding_blob, creation_time, creation_camera, total_appearances, update_count = row
                
                identity = GlobalIdentity(
                    global_id=global_id,
                    gallery_embedding=pickle.loads(embedding_blob),
                    creation_time=creation_time,
                    creation_camera=creation_camera,
                    total_appearances=total_appearances,
                    update_count=update_count
                )
                
                self.global_identities[global_id] = identity
            
            conn.close()
            
            logger.info(f"State loaded: {len(self.global_identities)} identities, next_id={self.next_global_id}")
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def save_checkpoint(self):
        """Manual checkpoint save"""
        self._save_state()
    
    def get_statistics(self) -> dict:
        """Get statistics for logging"""
        return {
            'next_global_id': self.next_global_id,
            'total_identities': len(self.global_identities),
            'active_tracks': {cam: len(tracks) for cam, tracks in self.active_tracks.items()},
            'cooldown_tracks': {cam: len(tracks) for cam, tracks in self.cooldown_tracks.items()},
            'pending_matches': len(self.pending_matches)
        }
