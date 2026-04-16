import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaissEngine:
    """
    Trình quản lý FAISS Index hỗ trợ streaming (tự động chuyển từ Flat sang IVF khi đủ dữ liệu).
    Sử dụng Inner Product (Cosine similarity) để match L2-normalized embedding.
    """
    def __init__(self, dimension: int = 512, train_threshold: int = 1000, 
                 nlist: int = 100, use_gpu: bool = False):
        self.dimension = dimension
        self.train_threshold = train_threshold
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.metric = faiss.METRIC_INNER_PRODUCT
        
        # Bắt đầu với Flat Index (lưu trữ chính xác, không cần train)
        # Gói trong IndexIDMap để hỗ trợ add_with_ids
        flat_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap(flat_index)
        
        self.is_ivf = False
        
        # Buffer để chứa vector trước khi IVF được train
        self._flat_vectors = []
        self._flat_ids = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Đảm bảo vector được L2-normalize trước khi sử dụng Inner Product (cosine)"""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def _upgrade_to_ivf(self):
        """
        Nâng cấp từ IndexFlat sang IndexIVFFlat khi đủ lượng vector để train
        """
        logger.info(f"Đang nâng cấp FAISS Index sang IndexIVFFlat với nlist={self.nlist}")
        quantizer = faiss.IndexFlatIP(self.dimension)
        ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, self.metric)
        
        # Cấu hình tuỳ chọn GPU nếu được yêu cầu
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                ivf_index = faiss.index_cpu_to_gpu(res, 0, ivf_index)
                logger.info("Đã chuyển IVF Index sang GPU.")
            except Exception as e:
                logger.error(f"Khởi tạo GPU FAISS lỗi, rớt về CPU: {e}")
                self.use_gpu = False

        # Train với các vector đã tích lũy
        train_data = np.vstack(self._flat_vectors)
        ivf_index.train(train_data)
        
        # Thêm toàn bộ dữ liệu cũ vào index mới
        # faiss.IndexIVFFlat (hoặc bản GPU) đều hỗ trợ add_with_ids khi đã được bọc đúng
        ivf_index.add_with_ids(train_data, np.array(self._flat_ids, dtype=np.int64))
        
        # Trỏ tham chiếu sang index mới
        self.index = ivf_index
        self.is_ivf = True
        
        # Dọn dẹp buffer RAM
        self._flat_vectors = None
        self._flat_ids = None
        logger.info("Đã hoàn tất nâng cấp lên IndexIVFFlat!")

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        """Thêm vector mới vào DB, kèm the id tĩnh do manager chỉ định"""
        norm_vectors = self._normalize(vectors.copy())
        
        self.index.add_with_ids(norm_vectors, ids)

        if not self.is_ivf:
            # Lưu lại vào buffer để dành train IVF
            self._flat_vectors.append(norm_vectors)
            self._flat_ids.extend(ids.tolist())
            
            # Kiểm tra xem đã đủ điều kiện upgrade (threshold) chưa
            if self.index.ntotal >= self.train_threshold:
                self._upgrade_to_ivf()

    def search(self, query_vectors: np.ndarray, k: int = 50, nprobe: int = 8):
        """Truy vấn Top-K Vector tương tự"""
        if self.index.ntotal == 0:
            return np.empty((query_vectors.shape[0], 0), dtype=np.float32), \
                   np.empty((query_vectors.shape[0], 0), dtype=np.int64)
                   
        if self.is_ivf:
            # Tuỳ chỉnh độ sâu dò tìm (trade-off giữa speed và accuracy)
            self.index.nprobe = nprobe
            
        norm_vectors = self._normalize(query_vectors.copy())
        distances, indices = self.index.search(norm_vectors, k)
        return distances, indices

    def remove_ids(self, ids_to_remove: np.ndarray) -> int:
        """
        Xóa các vector khỏi index dựa vào ID đã gán.
        Trả về số lượng ID đã xóa thành công.
        Lưu ý: FAISS remove_ids chỉ hoạt động trên CPU, nếu đang dùng GPU, 
        cần có cơ chế mapping chuyển về CPU rồi đẩy lại lên, hoặc soft-remove qua logic Metadata.
        """
        if self.use_gpu:
            logger.warning("FAISS GPU không support remove_ids native. Khuyên dùng soft-remove tại manager layer.")
            return 0
            
        try:
            # Native array in newer faiss versions
            sel = faiss.IDSelectorArray(ids_to_remove.tolist())
        except AttributeError:
            try:
                # Cấu trúc C++ swig (các bản faiss cũ hơn)
                sel = faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove))
            except AttributeError:
                sel = faiss.IDSelectorBatch(ids_to_remove)
                
        num_removed = self.index.remove_ids(sel)
        return num_removed

    @property
    def total_vectors(self):
        return self.index.ntotal
