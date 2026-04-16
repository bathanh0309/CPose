import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import threading
import logging

from .indexers import FaissEngine

logger = logging.getLogger(__name__)

@dataclass
class VectorMetadata:
    faiss_id: int
    global_id: str
    camera_id: str
    timestamp: float
    quality_score: float
    # Extra data placeholder để gắn thêm pose track_id nếu cần map lại
    extra_data: Dict[str, Any] = field(default_factory=dict)

class VectorDBManager:
    """
    Quản lý kiến trúc Vector DB kết hợp giữa FAISS Index Engine và In-memory Metadata (dict).
    Cơ chế Thread-safe tích hợp sẵn cho streaming.
    """
    def __init__(self, dimension: int = 512, train_threshold: int = 1000, use_gpu: bool = False):
        
        # Engine xử lý Vector Low-level
        self.engine = FaissEngine(
            dimension=dimension,
            train_threshold=train_threshold,
            nlist=max(10, int(np.sqrt(train_threshold) * 4)), # Auto compute nlist 
            use_gpu=use_gpu
        )
        
        # Primary store và Inverted Index
        self.metadata_store: Dict[int, VectorMetadata] = {}
        self.global_id_index: Dict[str, List[int]] = {}  # global_id -> list of faiss_ids
        
        self.next_faiss_id = 0
        self.lock = threading.RLock() # Thread-safe

    def insert(self, embedding: np.ndarray, global_id: str, camera_id: str, 
               timestamp: float, quality_score: float, extra_data: dict = None) -> int:
        """Thêm một embedding kèm metadata vào hệ thống."""
        with self.lock:
            faiss_id = self.next_faiss_id
            self.next_faiss_id += 1
            
            # 1. Chèn vào FAISS
            ids_arr = np.array([faiss_id], dtype=np.int64)
            vec = embedding.reshape(1, -1)
            self.engine.add(vec, ids_arr)
            
            # 2. Sinh và lưu Metadata
            meta = VectorMetadata(
                faiss_id=faiss_id,
                global_id=global_id,
                camera_id=camera_id,
                timestamp=timestamp,
                quality_score=quality_score,
                extra_data=extra_data or {}
            )
            self.metadata_store[faiss_id] = meta
            
            # 3. Chèn vào logic index ngược (dùng cho soft-remove, group-based processing sau này)
            if global_id not in self.global_id_index:
                self.global_id_index[global_id] = []
            self.global_id_index[global_id].append(faiss_id)
            
            return faiss_id

    def search(self, query_embedding: np.ndarray, k: int = 50, nprobe: int = 8) -> List[Tuple[float, VectorMetadata]]:
        """
        Tìm kiếm Top-K vector giống nhất.
        Trả về danh sách các tuple `(score, VectorMetadata)`.
        Score tiến gần 1.0 nghĩa là rất giống (Cosine Similarity).
        """
        with self.lock:
            vec = query_embedding.reshape(1, -1)
            distances, indices = self.engine.search(vec, k=k, nprobe=nprobe)
            
            results = []
            if len(distances) > 0 and len(distances[0]) > 0:
                for dist, faiss_id in zip(distances[0], indices[0]):
                    # Check Soft-remove hoặc Null ID
                    if faiss_id != -1 and faiss_id in self.metadata_store:
                        # Ép kiểu an toàn để jsonify khi truyền qua web socket
                        results.append((float(dist), self.metadata_store[faiss_id]))
                        
            return results

    def expire_old_vectors(self, max_age_seconds: float):
        """
        Cơ chế TTL (Online Expire): Dọn dẹp các vector cũ hơn thời gian quy định nhằm giảm tải IVF
        """
        current_time = time.time()
        ids_to_remove = []
        
        with self.lock:
            for faiss_id, meta in list(self.metadata_store.items()):
                if current_time - meta.timestamp > max_age_seconds:
                    ids_to_remove.append(faiss_id)
            
            if ids_to_remove:
                # 1. Hard-remove khỏi FAISS Index
                if not self.engine.use_gpu:
                    arr_ids = np.array(ids_to_remove, dtype=np.int64)
                    self.engine.remove_ids(arr_ids)
                
                # Chú ý: Nếu dùng GPU thì buộc sử dụng Soft-remove. 
                # Ta xóa ở dict metadata_store, khi search logic sẽ tự động ignore ID mất meta.
                
                # 2. Xóa Data
                for faiss_id in ids_to_remove:
                    meta = self.metadata_store.pop(faiss_id, None)
                    if meta:
                        try:
                            self.global_id_index[meta.global_id].remove(faiss_id)
                            # Remove root key nếu list đã rỗng
                            if not self.global_id_index[meta.global_id]:
                                del self.global_id_index[meta.global_id]
                        except ValueError:
                            pass
                            
                logger.info(f"[VectorDB] Expired {len(ids_to_remove)} vectors (Age > {max_age_seconds}s).")
