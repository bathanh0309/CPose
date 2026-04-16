import numpy as np
from typing import Dict, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from .manager import VectorMetadata

logger = logging.getLogger(__name__)

@dataclass
class STConstraint:
    min_transit_time: float  # Thời gian tối thiểu (giây) theo vật lý để di chuyển
    avg_transit_time: float  # Thời gian trung bình (có thể dùng để tuning distribution)

class SpatialTemporalFilter:
    """
    Bộ lọc Không gian - Thời gian (st-ReID) cho hệ thống multi-camera.
    Kiểm tra tính hợp lý vật lý của các ID qua lại giữa từng cụm Camera dựa trên Topology.
    Tích hợp hàm Logistic Smoothing thay vì loại bỏ cứng (hard-cutoff) để tránh False Negative.
    """
    def __init__(self):
        # Ma trận Topology: (camera_A, camera_B) -> Giới hạn vật lý
        self.topology: Dict[Tuple[str, str], STConstraint] = {}
        
    def add_path(self, cam_a: str, cam_b: str, min_time: float, avg_time: float, bidirectional: bool = True):
        """Khởi tạo một tuyến đường vật lý giữa 2 camera và đo lường khoảng thời gian."""
        self.topology[(cam_a, cam_b)] = STConstraint(min_time, avg_time)
        if bidirectional:
            self.topology[(cam_b, cam_a)] = STConstraint(min_time, avg_time)
            logger.info(f"[ST-Filter] Đăng ký Topology 2 chiều Cam {cam_a} <-> Cam {cam_b}: min={min_time}s")
        else:
            logger.info(f"[ST-Filter] Đăng ký Topology 1 chiều Cam {cam_a} -> Cam {cam_b}: min={min_time}s")

    def calculate_st_probability(self, query_cam: str, result_cam: str, delta_time: float) -> float:
        """
        Tính toán Xác suất Khả thi (ST-Probability) dựa theo thời gian lệch chuẩn.
        Sử dụng phân phối Logistic (Sigmoid transform) để tạo độ mượt.
        """
        # Nếu nằm chung 1 góc camera (phát hiện liên tiếp cục bộ) -> Khả thi 100%
        if query_cam == result_cam:
            return 1.0
            
        path = (result_cam, query_cam) # Từ quá khứ chạy đến hiện tại
        
        # Nếu không có topology (2 cam chẳng liên quan vật lý / ngoại vùng chéo)
        # Hoặc hệ thống chưa setup đủ bản đồ.
        if path not in self.topology:
            return 0.5  # Neutral point
            
        constraint = self.topology[path]
        
        # Dùng Logistic Function để biến 'hard limit' thành 'soft re-ranking'
        # P_st = 1 / (1 + exp(-k * (dt - min_t)))
        # - dt == min_t -> 50% khả năng
        # - dt > min_t (đi chậm/đúng luật) -> 90%++ khả năng
        # - dt < min_t (dịch chuyển tức thời / flash speed) -> Tụt thẳng dốc về 0%
        
        k = 3.0  # Độ dốc penalty (điều chỉnh độ gay gắt của filter)
        p = 1.0 / (1.0 + np.exp(-k * (delta_time - constraint.min_transit_time)))
        
        # Tránh đưa về số 0 hoàn toàn (đề phòng Server Clock drift / NTP lệch mạng)
        # Đặt mức sàn tối thiểu là 5% (0.05)
        return max(0.05, float(p))

    def refine_scores(self, query_cam: str, query_time: float, 
                      faiss_results: List[Tuple[float, 'VectorMetadata']],
                      alpha: float = 0.5) -> List[Tuple[float, 'VectorMetadata']]:
        """
        Re-ranking danh sách trả về từ FAISS bằng quy tắc ST-ReID.
        :param alpha: Trọng số định hướng (0.0 thuần theo AI Đặc trứng, 1.0 thuần theo Khoảng cách trạm).
        Thường để 0.5 hoặc 0.6.
        """
        refined_results = []
        for feat_score, meta in faiss_results:
            # dt là độ chênh lệch thời gian từ vector quá khứ tới frame suy diễn hiện tại
            dt = abs(query_time - meta.timestamp)
            
            # Chấm điểm thời gian-không gian
            st_prob = self.calculate_st_probability(query_cam, meta.camera_id, dt)
            
            # Linear Interpolation: (1 - α) * Sự giống nhau mặt hình học AI + α * Sự hợp lý vật lý ST
            # Cosine similarity thường lân cận quanh [0..1] cho ảnh giống nhau
            final_score = (1.0 - alpha) * feat_score + alpha * st_prob
            
            # Ghi đè tuple mới (do tupler immutable)
            refined_results.append((final_score, meta))
            
        # Sắp xếp lại danh sách ứng viên (Candidate) theo final rank chốt hạ
        refined_results.sort(key=lambda x: x[0], reverse=True)
        return refined_results
