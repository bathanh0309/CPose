import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import threading
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class TrackCentroid:
    """Đại diện cho luồng dữ liệu 1 track của 1 người đang di chuyển trong 1 camera"""
    track_id: str           # VD: f"{camera_id}_{local_track_id}"
    global_id: str          # None nếu chưa match được vào hệ thống, có ID nếu đã match
    centroid: np.ndarray    # Vector trọng tâm (tích lũy)
    num_updates: int        # Số lượng frame/vector đã chèn vào
    last_timestamp: float   # Thời điểm frame cuối
    last_quality: float     # Điểm chất lượng Bounding box của frame cuối
    is_flushed: bool = False # Đã chèn vào FAISS chưa?

class EMACentralizer:
    """
    Bộ quản lý Exponential Moving Average Vectors.
    Gộp (merge) nhiều frames của 1 người trong 1 camera thành 1 Vector đại diện tốt nhất (Centroid).
    """
    def __init__(self, alpha: float = 0.2, min_updates_to_flush: int = 5):
        """
        - alpha: Hệ số lấy mẫu mới (0 < alpha <= 1). 
            alpha=0.2 nghĩa là vector mới chiếm 20% trọng số, vector cũ giữ 80%.
        - min_updates_to_flush: Độ dài track tối thiểu để vector được coi là ổn định 
            và có giá trị trích xuất (tránh track nhiễu sinh ra do lỗi DeepSORT).
        """
        self.alpha = alpha
        self.min_updates_to_flush = min_updates_to_flush
        self.active_tracks: Dict[str, TrackCentroid] = {}
        self.lock = threading.RLock()
        
    def update(self, track_id: str, new_embedding: np.ndarray, 
               timestamp: float, quality_score: float, global_id: str = None) -> TrackCentroid:
        """
        Đưa 1 feature vector mới của frame hiện tại vào track.
        EMA cập nhật: centroid_mới = alpha_động * new_vec + (1 - alpha_động) * centroid_cũ.
        alpha_động được scale theo quality_score để giảm nhiễu từ ảnh mờ.
        """
        with self.lock:
            # Đảm bảo shape 1D float32
            vec = new_embedding.flatten().astype(np.float32)
            
            if track_id not in self.active_tracks:
                # Khởi tạo
                self.active_tracks[track_id] = TrackCentroid(
                    track_id=track_id,
                    global_id=global_id,
                    centroid=vec.copy(),
                    num_updates=1,
                    last_timestamp=timestamp,
                    last_quality=quality_score
                )
            else:
                # Cập nhật lũy tiến với EMA
                track = self.active_tracks[track_id]
                
                # Dynamic alpha: Nếu ảnh rất nét (quality cao), alpha giữ nguyên.
                # Nếu ảnh mờ (quality thấp), vector này ít tác động vào centroid gốc hơn.
                dynamic_alpha = self.alpha * (quality_score + 0.1) 
                dynamic_alpha = min(max(dynamic_alpha, 0.01), 0.99)
                
                new_centroid = dynamic_alpha * vec + (1.0 - dynamic_alpha) * track.centroid
                
                track.centroid = new_centroid
                track.num_updates += 1
                track.last_timestamp = max(track.last_timestamp, timestamp)
                track.last_quality = quality_score
                if global_id:
                    track.global_id = global_id
            
            return self.active_tracks[track_id]

    def get_normalized_centroid(self, track_id: str) -> np.ndarray:
        """
        Lấy centroid hiện tại và tự động L2-normalize. 
        Khi cộng 2 vector theo EMA, độ dài L2 sẽ bị lệch khỏi 1.0, 
        do đó ta cần trực tiếp normalize lại khi lấy ra để FAISS xài cosine.
        """
        with self.lock:
            if track_id not in self.active_tracks:
                return None
            
            c = self.active_tracks[track_id].centroid.copy()
            norm = np.linalg.norm(c)
            if norm > 0:
                c = c / norm
            return c
            
    def is_track_ready(self, track_id: str) -> bool:
        """Kiểm tra track đã đủ dài/đủ ổn định để search FAISS hoặc chèn vào DB chưa"""
        with self.lock:
            if track_id not in self.active_tracks:
                return False
            track = self.active_tracks[track_id]
            return track.num_updates >= self.min_updates_to_flush

    def pop_expired_tracks(self, max_age_seconds: float) -> List[TrackCentroid]:
        """
        Dọn dẹp các local active_tracks đã biến mất khỏi camera quá `max_age_seconds`.
        Thường gọi mỗi chu kì (VD mỗi 3-5 giây).
        """
        current_timestamp = time.time()
        expired = []
        with self.lock:
            for tid, tdata in list(self.active_tracks.items()):
                if current_timestamp - tdata.last_timestamp > max_age_seconds:
                    expired.append(tdata)
                    del self.active_tracks[tid]
        
        if expired:
            logger.info(f"[EMA] Dọn dẹp {len(expired)} local tracks đã rời camera.")
            
        return expired
