"""
Enhanced Color + Shape ReID with Temporal Voting
=================================================
- HSV color histogram (3 parts: head, body, legs)
- Value channel for brightness discrimination
- Hu Moments for shape
- Temporal voting (confirm_frames) before creating new ID
- Hungarian assignment for multi-person matching
"""
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class EnhancedReID:
    """Enhanced ReID using HSV + Shape features."""
    
    def extract(self, img_crop):
        """Extract features from person crop."""
        if img_crop is None or img_crop.size == 0:
            return None
        if img_crop.shape[0] < 30 or img_crop.shape[1] < 15:
            return None
        
        try:
            hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            h = img_crop.shape[0]
            
            # 3 parts: head (20%), body (50%), legs (30%)
            parts_hsv = [
                hsv[:int(h*0.2), :],           # Head
                hsv[int(h*0.2):int(h*0.7), :], # Body (largest - most important)
                hsv[int(h*0.7):, :]            # Legs
            ]
            
            features = []
            
            # Color features (H, S, V channels with more bins)
            for part in parts_hsv:
                if part.size == 0:
                    features.extend([0] * 56)  # 24 + 16 + 16
                    continue
                
                # H channel (24 bins for finer color discrimination)
                h_hist = cv2.calcHist([part], [0], None, [24], [0, 180])
                h_hist = cv2.normalize(h_hist, h_hist).flatten()
                
                # S channel (16 bins)
                s_hist = cv2.calcHist([part], [1], None, [16], [0, 256])
                s_hist = cv2.normalize(s_hist, s_hist).flatten()
                
                # V channel (16 bins for brightness - important for cross-camera)
                v_hist = cv2.calcHist([part], [2], None, [16], [0, 256])
                v_hist = cv2.normalize(v_hist, v_hist).flatten()
                
                features.extend(h_hist)
                features.extend(s_hist)
                features.extend(v_hist)
            
            # Shape features using Hu Moments (scale & rotation invariant)
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform to reduce range
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            # Normalize
            hu_moments = hu_moments / (np.linalg.norm(hu_moments) + 1e-10)
            features.extend(hu_moments)
            
            # Aspect ratio
            aspect = img_crop.shape[1] / img_crop.shape[0]
            features.append(aspect)
            
            return np.array(features)
        except Exception as e:
            return None


class MasterSlaveReIDDB:
    """Master-Slave ReID Database with Temporal Voting."""
    
    def __init__(self, reid_threshold=0.65, max_features=100, confirm_frames=3):
        self.reid = EnhancedReID()
        self.persons = {}  # global_id -> {features: [], first_cam: str, cameras: set}
        self.next_id = 1
        self.reid_threshold = reid_threshold
        self.max_features = max_features
        self.new_ids_allowed = True
        self.confirm_frames = confirm_frames  # Temporal voting
        
        # Temporal voting buffers
        self.pending_ids = {}  # track_id -> {features: [], votes: int, best_match: None}
        self.track_to_global = {}  # track_id -> global_id (confirmed)
    
    def _get_similarity(self, features, gid):
        """Get similarity between features and a global ID."""
        if gid not in self.persons:
            return 0.0
        
        info = self.persons[gid]
        sims = []
        for stored_feat in info['features']:
            try:
                sim = 1 - cosine(features, stored_feat)
                if not np.isnan(sim):
                    sims.append(sim)
            except:
                continue
        
        if not sims:
            return 0.0
        
        # TOP-K averaging
        sims.sort(reverse=True)
        top_k = min(5, len(sims))
        return np.mean(sims[:top_k])
    
    def register_new(self, person_crop, cam_id, track_id=None):
        """Register new person with temporal voting (ONLY Master cam)."""
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        # If track_id provided, use temporal voting
        if track_id is not None:
            # Already confirmed?
            if track_id in self.track_to_global:
                gid = self.track_to_global[track_id]
                self._update_features(gid, features)
                return gid
            
            # Check if matches existing
            best_id, best_sim = self._find_best_match(features)
            
            if best_id and best_sim > self.reid_threshold:
                # Temporal voting for matching
                if track_id not in self.pending_ids:
                    self.pending_ids[track_id] = {'votes': 0, 'best_match': best_id, 'features': [features]}
                else:
                    self.pending_ids[track_id]['votes'] += 1
                    self.pending_ids[track_id]['features'].append(features)
                    # Keep best match updated
                    if best_sim > self._get_similarity(self.pending_ids[track_id]['features'][0], 
                                                       self.pending_ids[track_id]['best_match']):
                        self.pending_ids[track_id]['best_match'] = best_id
                
                # Confirmed?
                if self.pending_ids[track_id]['votes'] >= self.confirm_frames:
                    gid = self.pending_ids[track_id]['best_match']
                    self.track_to_global[track_id] = gid
                    # Add all pending features
                    for feat in self.pending_ids[track_id]['features']:
                        self._update_features(gid, feat)
                    del self.pending_ids[track_id]
                    if cam_id not in self.persons[gid]['cameras']:
                        self.persons[gid]['cameras'].add(cam_id)
                        print(f"    G-ID {gid} confirmed in {cam_id.upper()} (temporal voting)")
                    return gid
                return None  # Still pending
            else:
                # New person - temporal voting before creating
                if track_id not in self.pending_ids:
                    self.pending_ids[track_id] = {'votes': 1, 'best_match': None, 'features': [features]}
                else:
                    self.pending_ids[track_id]['votes'] += 1
                    self.pending_ids[track_id]['features'].append(features)
                
                # Confirmed as new person?
                if self.pending_ids[track_id]['votes'] >= self.confirm_frames:
                    new_id = self._create_new_id(features, cam_id)
                    self.track_to_global[track_id] = new_id
                    # Add all pending features
                    for feat in self.pending_ids[track_id]['features'][1:]:
                        self._update_features(new_id, feat)
                    del self.pending_ids[track_id]
                    return new_id
                return None  # Still pending
        
        # No track_id - immediate creation (legacy behavior)
        return self._create_new_id(features, cam_id)
    
    def _create_new_id(self, features, cam_id):
        """Actually create new ID."""
        new_id = self.next_id
        self.next_id += 1
        
        self.persons[new_id] = {
            'features': [features],
            'first_cam': cam_id,
            'cameras': {cam_id}
        }
        
        print(f"    G-ID {new_id} created in {cam_id.upper()}")
        return new_id
    
    def _update_features(self, gid, features):
        """Update features for existing ID."""
        if gid in self.persons and len(self.persons[gid]['features']) < self.max_features:
            self.persons[gid]['features'].append(features)
    
    def _find_best_match(self, features):
        """Find best matching global ID."""
        best_id = None
        best_sim = 0
        
        for gid in self.persons:
            sim = self._get_similarity(features, gid)
            if sim > best_sim:
                best_sim = sim
                best_id = gid
        
        return best_id, best_sim
    
    def match_only(self, person_crop, cam_id, track_id=None):
        """Match against existing IDs with temporal voting."""
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        # Already confirmed?
        if track_id is not None and track_id in self.track_to_global:
            gid = self.track_to_global[track_id]
            self._update_features(gid, features)
            return gid
        
        best_id, best_sim = self._find_best_match(features)
        
        if best_id is None or best_sim < self.reid_threshold:
            return None
        
        # Temporal voting for slave cameras
        if track_id is not None:
            if track_id not in self.pending_ids:
                self.pending_ids[track_id] = {'votes': 1, 'best_match': best_id, 'features': [features]}
            else:
                self.pending_ids[track_id]['votes'] += 1
                self.pending_ids[track_id]['features'].append(features)
                # Update best match if better
                if best_sim > self._get_similarity(features, self.pending_ids[track_id]['best_match']):
                    self.pending_ids[track_id]['best_match'] = best_id
            
            # Confirmed?
            if self.pending_ids[track_id]['votes'] >= self.confirm_frames:
                gid = self.pending_ids[track_id]['best_match']
                self.track_to_global[track_id] = gid
                for feat in self.pending_ids[track_id]['features']:
                    self._update_features(gid, feat)
                del self.pending_ids[track_id]
                
                if cam_id not in self.persons[gid]['cameras']:
                    self.persons[gid]['cameras'].add(cam_id)
                    print(f"    G-ID {gid} matched in {cam_id.upper()} (sim={best_sim:.3f})")
                return gid
            return None  # Still pending
        
        # No track_id - immediate match (legacy)
        if cam_id not in self.persons[best_id]['cameras']:
            self.persons[best_id]['cameras'].add(cam_id)
            print(f"    G-ID {best_id} matched in {cam_id.upper()} (sim={best_sim:.3f})")
        self._update_features(best_id, features)
        return best_id
    
    def match_exclusive(self, person_crop, cam_id, exclude_ids=None, track_id=None):
        """Match against existing IDs, excluding some IDs."""
        if exclude_ids is None:
            exclude_ids = set()
        
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        best_id = None
        best_sim = 0
        
        for gid in self.persons:
            if gid in exclude_ids:
                continue
            sim = self._get_similarity(features, gid)
            if sim > self.reid_threshold and sim > best_sim:
                best_sim = sim
                best_id = gid
        
        if best_id:
            if cam_id not in self.persons[best_id]['cameras']:
                self.persons[best_id]['cameras'].add(cam_id)
                print(f"    G-ID {best_id} matched in {cam_id.upper()} (sim={best_sim:.3f})")
            self._update_features(best_id, features)
            return best_id
        
        return None
    
    def hungarian_assign(self, crops_with_tracks, cam_id, is_master=False):
        """Use Hungarian algorithm to optimally assign multiple people at once.
        
        Args:
            crops_with_tracks: list of (person_crop, track_id) tuples
            cam_id: camera ID
            is_master: if True, can create new IDs
        
        Returns:
            dict: track_id -> global_id (or None if no match)
        """
        if not crops_with_tracks:
            return {}
        
        # Extract all features
        valid_items = []
        for crop, track_id in crops_with_tracks:
            # Check if already confirmed
            if track_id in self.track_to_global:
                continue  # Skip, already assigned
            feat = self.reid.extract(crop)
            if feat is not None:
                valid_items.append((crop, track_id, feat))
        
        if not valid_items:
            return {tid: self.track_to_global.get(tid) for _, tid in crops_with_tracks}
        
        # Build cost matrix
        n_tracks = len(valid_items)
        n_gids = len(self.persons)
        
        if n_gids == 0:
            # No existing IDs - create new ones if master
            result = {}
            for crop, track_id, feat in valid_items:
                if is_master:
                    gid = self.register_new(crop, cam_id, track_id)
                    result[track_id] = gid
                else:
                    result[track_id] = None
            return result
        
        gid_list = list(self.persons.keys())
        cost_matrix = np.zeros((n_tracks, n_gids))
        
        for i, (crop, track_id, feat) in enumerate(valid_items):
            for j, gid in enumerate(gid_list):
                sim = self._get_similarity(feat, gid)
                cost_matrix[i, j] = 1 - sim  # Convert to cost
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        result = {}
        used_gids = set()
        
        for i, j in zip(row_ind, col_ind):
            crop, track_id, feat = valid_items[i]
            gid = gid_list[j]
            sim = 1 - cost_matrix[i, j]
            
            if sim >= self.reid_threshold:
                result[track_id] = self.match_only(crop, cam_id, track_id)
                used_gids.add(gid)
            elif is_master:
                result[track_id] = self.register_new(crop, cam_id, track_id)
            else:
                result[track_id] = None
        
        # Add already confirmed IDs
        for _, tid in crops_with_tracks:
            if tid in self.track_to_global and tid not in result:
                result[tid] = self.track_to_global[tid]
        
        return result
    
    def reset_track_mapping(self):
        """Reset track-to-global mapping (call when switching cameras)."""
        self.pending_ids.clear()
        self.track_to_global.clear()
    
    def update_features(self, global_id, person_crop):
        """Update features for existing ID (legacy API)."""
        if global_id not in self.persons:
            return
        
        if len(self.persons[global_id]['features']) >= self.max_features:
            return
        
        features = self.reid.extract(person_crop)
        if features is not None:
            self.persons[global_id]['features'].append(features)
    
    def summary(self):
        """Print summary."""
        print(f"\n{'='*60}")
        print(f" GLOBAL ID SUMMARY")
        print(f"{'='*60}")
        print(f"Total Persons: {len(self.persons)}")
        for gid, info in sorted(self.persons.items()):
            cams = "  ".join(sorted(info['cameras']))
            feat_count = len(info['features'])
            print(f"  Person-{gid}: {cams} ({feat_count} features)")
