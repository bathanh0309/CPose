"""
Enhanced Color + Shape ReID
===========================
- HSV color histogram (3 parts: head, body, legs)
- Value channel for brightness discrimination
- Hu Moments for shape
- Stricter matching for cross-camera
"""
import cv2
import numpy as np
from scipy.spatial.distance import cosine


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
    """Master-Slave ReID Database with strict cross-camera matching."""
    
    def __init__(self, reid_threshold=0.65, max_features=1000):
        self.reid = EnhancedReID()
        self.persons = {}  # global_id -> {features: [], first_cam: str, cameras: set}
        self.next_id = 1
        self.reid_threshold = reid_threshold  # Giảm xuống 0.65 để match tốt hơn
        self.max_features = max_features  # 1000 features
        self.new_ids_allowed = True
    
    def register_new(self, person_crop, cam_id):
        """Register new person (ONLY Master cam)."""
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        new_id = self.next_id
        self.next_id += 1
        
        self.persons[new_id] = {
            'features': [features],
            'first_cam': cam_id,
            'cameras': {cam_id}
        }
        
        print(f"    G-ID {new_id} created in {cam_id.upper()}")
        return new_id
    
    def match_only(self, person_crop, cam_id):
        """Match against existing IDs."""
        return self.match_exclusive(person_crop, cam_id, exclude_ids=set())
    
    def match_exclusive(self, person_crop, cam_id, exclude_ids=None):
        """Match against existing IDs, excluding some IDs.
        
        Uses TOP-K averaging for more stable matching:
        - Compare with ALL stored features
        - Use top-5 similarities for final score
        - Stricter threshold for cross-camera matching
        """
        if exclude_ids is None:
            exclude_ids = set()
        
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        best_id = None
        best_sim = 0
        
        # Cross-camera requires higher threshold
        threshold = self.reid_threshold
        
        for gid, info in self.persons.items():
            if gid in exclude_ids:
                continue
            
            # Compare with ALL stored features
            sims = []
            for stored_feat in info['features']:
                try:
                    sim = 1 - cosine(features, stored_feat)
                    if not np.isnan(sim):
                        sims.append(sim)
                except:
                    continue
            
            if not sims:
                continue
            
            # Use TOP-K averaging (more robust than just mean)
            sims.sort(reverse=True)
            top_k = min(5, len(sims))
            avg_sim = np.mean(sims[:top_k])
            
            # Also check max similarity
            max_sim = sims[0]
            
            # Both avg and max must be above threshold
            if avg_sim > threshold and max_sim > threshold + 0.05:
                if avg_sim > best_sim:
                    best_id = gid
                    best_sim = avg_sim
        
        if best_id:
            # Update features (limit frequency to avoid noise)
            if len(self.persons[best_id]['features']) < self.max_features:
                self.persons[best_id]['features'].append(features)
            
            # Track cameras
            if cam_id not in self.persons[best_id]['cameras']:
                self.persons[best_id]['cameras'].add(cam_id)
                print(f"    G-ID {best_id} matched in {cam_id.upper()} (sim={best_sim:.3f})")
            
            return best_id
        
        return None
    
    def update_features(self, global_id, person_crop):
        """Update features for existing ID."""
        if global_id not in self.persons:
            return
        
        # Only update occasionally to keep representative features
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
