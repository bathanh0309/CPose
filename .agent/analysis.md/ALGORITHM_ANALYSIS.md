# HAVEN - Phân Tích Hệ Thống DSA & Kiến Trúc

## 📋 Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Kiến Trúc Tổng Thể](#2-kiến-trúc-tổng-thể)
3. [Pipeline Xử Lý](#3-pipeline-xử-lý)
4. [Cấu Trúc Dữ Liệu](#4-cấu-trúc-dữ-liệu)
5. [Thuật Toán Chính](#5-thuật-toán-chính)
6. [Luồng Dữ Liệu](#6-luồng-dữ-liệu)
7. [Module Chi Tiết](#7-module-chi-tiết)
8. [Phân Tích Độ Phức Tạp](#8-phân-tích-độ-phức-tạp)

---

## 1. Tổng Quan Hệ Thống

### 1.1 Mục Đích
HAVEN (Healthcare Activity Video Evaluation Network) là hệ thống **Multi-Camera Person Re-Identification** kết hợp:
- **Nhận diện người** (YOLO Detection)
- **Theo dõi đa camera** (ByteTrack + Global ID)
- **Phân tích tư thế** (Pose Estimation)
- **Phát hiện hoạt động** (ADL - Activities of Daily Living)

### 1.2 Công Nghệ Sử Dụng
| Thành phần | Công nghệ |
|------------|-----------|
| Detection | YOLOv8/YOLOv11 |
| Tracking | ByteTrack |
| Pose | YOLOv8-Pose (17 keypoints) |
| ReID | HSV Histogram + Hu Moments |
| Storage | SQLite + NumPy Memmap |
| Vector Search | FAISS |
| Backend | Python + FastAPI |
| Frontend | HTML/JS + WebSocket |

---

## 2. Kiến Trúc Tổng Thể

### 2.1 Sơ Đồ Kiến Trúc Master-Slave

```
┌─────────────────────────────────────────────────────────────────┐
│                    HAVEN Multi-Camera System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Cam 1   │    │  Cam 2   │    │  Cam 3   │    │  Cam 4   │ │
│   │ (MASTER) │    │ (MASTER) │    │ (SLAVE)  │    │ (SLAVE)  │ │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘ │
│        │               │               │               │        │
│        └───────────────┴───────┬───────┴───────────────┘        │
│                                │                                 │
│                    ┌───────────▼───────────┐                    │
│                    │   SequentialRunner    │                    │
│                    │   (Orchestrator)      │                    │
│                    └───────────┬───────────┘                    │
│                                │                                 │
│        ┌───────────────────────┼───────────────────────┐        │
│        ▼                       ▼                       ▼        │
│  ┌──────────┐          ┌──────────────┐         ┌──────────┐   │
│  │   YOLO   │          │ GlobalID     │         │   ADL    │   │
│  │ Detector │          │ Manager      │         │  Engine  │   │
│  │ + Pose   │          │              │         │          │   │
│  └────┬─────┘          └──────┬───────┘         └────┬─────┘   │
│       │                       │                      │          │
│       │                ┌──────┴──────┐               │          │
│       │                ▼             ▼               │          │
│       │         ┌──────────┐  ┌──────────┐          │          │
│       │         │Persistence│  │ VectorDB │          │          │
│       │         │ (SQLite)  │  │ (FAISS)  │          │          │
│       │         └──────────┘  └──────────┘          │          │
│       │                                              │          │
│       └──────────────────────┬───────────────────────┘          │
│                              ▼                                   │
│                    ┌───────────────────┐                        │
│                    │  Output Video     │                        │
│                    │  + WebSocket API  │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Vai Trò Camera

| Camera | Vai trò | Quyền hạn |
|--------|---------|-----------|
| Master (Cam1, Cam2) | Tạo Global ID mới | CREATE + MATCH |
| Slave (Cam3, Cam4) | Chỉ matching | MATCH only |

---

## 3. Pipeline Xử Lý

### 3.1 Quy Trình Xử Lý Tuần Tự

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: VIDEO INPUT                                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ VideoCapture → Frame → Resize (640x480) → Preprocessing    ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ▼                                   │
│  Step 2: DETECTION (YOLO)                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ YOLO(frame) → [{bbox, conf, class, track_id, keypoints}]   ││
│  │ Filter: conf > 0.5, class == 'person'                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ▼                                   │
│  Step 3: TRACKING (ByteTrack)                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ByteTrack.update(detections) → [{track_id, bbox, age}]     ││
│  │ Maintain: track_id across frames                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ▼                                   │
│  Step 4: FEATURE EXTRACTION (ReID)                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ For each person:                                            ││
│  │   crop = frame[y1:y2, x1:x2]                                ││
│  │   embedding = HSV_histogram(168) + HuMoments(7) + ratio(1) ││
│  │   → 176-dim feature vector (L2-normalized)                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ▼                                   │
│  Step 5: GLOBAL ID ASSIGNMENT                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ If MASTER camera:                                           ││
│  │   → Match existing OR Create new GlobalID                   ││
│  │ If SLAVE camera:                                            ││
│  │   → Match existing OR Assign UNK                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ▼                                   │
│  Step 6: POSE & ADL ANALYSIS                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ keypoints → posture (standing/sitting/lying/unknown)       ││
│  │ FSM: state transitions → events (fall, bed_exit, etc.)     ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ▼                                   │
│  Step 7: OUTPUT                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Draw: bbox + skeleton + labels → video_output              ││
│  │ Save: CSV events, SQLite state, GIF/MP4                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Thời Gian Xử Lý (Ước Tính)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| YOLO Detection | 15-30ms | GPU: 10ms, CPU: 30ms |
| ByteTrack | 1-2ms | Lightweight |
| Feature Extraction | 5-10ms | Per person |
| Global ID Matching | 1-5ms | FAISS O(log N) |
| Pose Classification | 1ms | Rule-based |
| **Total per frame** | **~50ms** | ~20 FPS |

---

## 4. Cấu Trúc Dữ Liệu

### 4.1 Core Classes

#### 4.1.1 GlobalIDManager
```python
class GlobalIDManager:
    """Quản lý ID toàn cục cho multi-camera tracking."""
    
    # State
    master_camera: str              # Camera chủ (VD: 'cam2')
    slave_cameras: List[str]        # Cameras phụ ['cam3', 'cam4']
    
    # Tracking state
    match_history: Dict[int, Dict[int, int]]
    # Structure: {track_id: {global_id: vote_count}}
    # Dùng temporal voting để xác nhận match
    
    last_seen: Dict[int, float]     # {global_id: timestamp}
    unk_counter: Dict[str, int]     # {camera: unk_count}
    
    # External dependencies
    persistence: PersistenceManager  # SQLite storage
    vector_db: VectorDatabase        # FAISS index
    matcher: HybridMatcher           # Search + filter
```

#### 4.1.2 EnhancedReID
```python
class EnhancedReID:
    """Trích xuất đặc trưng ReID từ ảnh cắt người."""
    
    def extract(self, crop) -> np.ndarray:
        """
        Input: BGR image crop (person bounding box)
        Output: 176-dim normalized feature vector
        
        Components:
        - HSV histogram: 3 parts x 56 bins = 168 dims
        - Hu Moments: 7 dims  
        - Aspect ratio: 1 dim
        """
```

#### 4.1.3 MasterSlaveReIDDB
```python
class MasterSlaveReIDDB:
    """ReID Database với temporal voting."""
    
    # Gallery storage
    gallery: Dict[int, np.ndarray]  # {global_id: mean_features}
    gallery_samples: Dict[int, List[np.ndarray]]  # Multi-prototype
    
    # Voting state
    pending_tracks: Dict[int, Dict[int, int]]
    # {track_id: {global_id: vote_count}}
    
    # Mapping
    track_to_global: Dict[int, int]  # Local → Global
```

### 4.2 Database Schema (SQLite)

```sql
-- Bảng chính lưu GlobalID
CREATE TABLE global_ids (
    global_id INTEGER PRIMARY KEY,
    created_at TIMESTAMP,
    created_camera TEXT,
    is_active INTEGER DEFAULT 1,
    embedding_idx INTEGER,      -- Index trong memmap
    last_bbox TEXT,             -- JSON: [x1,y1,x2,y2]
    last_seen_camera TEXT,
    last_seen_time TIMESTAMP
);

-- Bảng theo dõi chuyển camera
CREATE TABLE camera_transitions (
    id INTEGER PRIMARY KEY,
    global_id INTEGER,
    from_camera TEXT,
    to_camera TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (global_id) REFERENCES global_ids(global_id)
);

-- Index để tối ưu query
CREATE INDEX idx_gid_camera ON global_ids(last_seen_camera);
CREATE INDEX idx_gid_active ON global_ids(is_active);
```

### 4.3 Vector Database (FAISS)

```python
class VectorDatabase:
    """FAISS-based ANN search."""
    
    # Index selection by size
    # N < 1000:      IndexFlatL2 (exact, O(N))
    # 1000 <= N < 10000: IndexHNSWFlat (O(log N))
    # N >= 10000:    IndexIVFFlat (O(sqrt(N)))
    
    index: faiss.Index
    global_ids: np.ndarray      # Mapping index → global_id
    embeddings: np.ndarray      # (N, 176) matrix
```

---

## 5. Thuật Toán Chính

### 5.1 Thuật Toán Gán Global ID

#### 5.1.1 Master Camera Logic
```python
def _assign_master(track_id, embedding, bbox, frame_time, quality):
    """
    MASTER camera: Có quyền tạo GlobalID mới.
    
    Algorithm:
    1. Search vector database cho candidates
    2. Two-threshold decision:
       - score >= strong_threshold (0.65): MATCH
       - score < weak_threshold (0.40): CREATE NEW
       - Giữa 2 ngưỡng: Temporal voting
    3. Update gallery với EMA
    """
    
    # Bước 1: Tìm ứng viên
    candidates = matcher.match(embedding, top_k=20)
    
    if len(candidates) == 0:
        # Gallery rỗng → Tạo ID đầu tiên
        return _create_new_id(embedding, bbox)
    
    best_match = candidates[0]
    score = best_match['score']
    global_id = best_match['global_id']
    
    # Bước 2: Two-threshold decision
    if score >= STRONG_THRESHOLD:  # 0.65
        # Strong match → Xác nhận qua voting
        match_history[track_id][global_id] += 1
        
        if match_history[track_id][global_id] >= CONFIRM_FRAMES:
            # Đã đủ votes → Confirmed
            return global_id, "CONFIRMED"
        else:
            return global_id, "PENDING"
    
    elif score < WEAK_THRESHOLD:  # 0.40
        # Definitely new person
        return _create_new_id(embedding, bbox)
    
    else:
        # Uncertain zone [0.40, 0.65]
        # Cần thêm frames để quyết định
        return None, "UNCERTAIN"
```

#### 5.1.2 Slave Camera Logic
```python
def _assign_slave(camera, track_id, embedding, bbox, frame_time):
    """
    SLAVE camera: CHỈ được matching, KHÔNG tạo ID mới.
    
    Algorithm:
    1. Search vector database
    2. Apply spatiotemporal filtering
    3. Match hoặc gán UNK
    """
    
    # Bước 1: Lấy candidates hợp lệ
    valid_ids = _get_valid_candidates(camera, frame_time)
    candidates = matcher.match(embedding, allowed_ids=valid_ids)
    
    if len(candidates) == 0:
        return _assign_unk(camera, track_id, bbox, "empty_gallery")
    
    best = candidates[0]
    
    # Bước 2: Check threshold
    if best['score'] >= STRONG_THRESHOLD:
        return best['global_id'], "MATCHED"
    
    # Không đủ điểm → UNK
    return _assign_unk(camera, track_id, bbox, "low_similarity")
```

### 5.2 Thuật Toán Feature Extraction

```python
def extract_features(crop):
    """
    Trích xuất 176-dim feature vector.
    
    Algorithm:
    1. Chia crop thành 3 phần: head, body, legs
    2. Mỗi phần: HSV histogram (H:16, S:16, V:24 bins)
    3. Tính Hu Moments cho shape
    4. Thêm aspect ratio
    5. L2 normalize
    """
    
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    features = []
    
    # 3 parts: head (0-33%), body (33-66%), legs (66-100%)
    parts = [
        hsv[0:h//3, :],
        hsv[h//3:2*h//3, :],
        hsv[2*h//3:, :]
    ]
    
    for part in parts:
        # H channel: 16 bins
        h_hist = cv2.calcHist([part], [0], None, [16], [0,180])
        # S channel: 16 bins
        s_hist = cv2.calcHist([part], [1], None, [16], [0,256])
        # V channel: 24 bins  
        v_hist = cv2.calcHist([part], [2], None, [24], [0,256])
        
        features.extend([h_hist, s_hist, v_hist])
    
    # Hu Moments (7 dims)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    features.append(hu)
    
    # Aspect ratio (1 dim)
    features.append([h / max(w, 1)])
    
    # Concatenate và normalize
    feature_vector = np.concatenate(features)
    feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-6)
    
    return feature_vector  # 176 dims
```

### 5.3 Thuật Toán Temporal Voting

```python
class TemporalVoting:
    """
    Vote-based ID confirmation để chống flicker.
    
    Ý tưởng:
    - Một match đơn lẻ có thể là false positive
    - Yêu cầu N frames liên tiếp match cùng ID
    - Reset votes khi switch sang ID khác
    """
    
    def __init__(self, confirm_frames=15):
        self.confirm_frames = confirm_frames
        self.votes = defaultdict(lambda: defaultdict(int))
        # {track_id: {global_id: count}}
    
    def vote(self, track_id, global_id):
        """Thêm 1 vote cho (track_id, global_id)."""
        self.votes[track_id][global_id] += 1
        return self.votes[track_id][global_id]
    
    def is_confirmed(self, track_id, global_id):
        """Check đã đủ votes chưa."""
        return self.votes[track_id][global_id] >= self.confirm_frames
    
    def get_best(self, track_id):
        """Lấy global_id có nhiều votes nhất."""
        if track_id not in self.votes:
            return None, 0
        votes = self.votes[track_id]
        best_id = max(votes, key=votes.get)
        return best_id, votes[best_id]
```

### 5.4 Thuật Toán Hungarian Assignment

```python
def hungarian_assign(crops_with_tracks, cam_id, is_master=False):
    """
    Optimal assignment multi-person → multi-ID.
    
    Algorithm: Hungarian (Kuhn-Munkres)
    Complexity: O(n³)
    
    Dùng khi: Có nhiều người trong 1 frame
    """
    
    n_detections = len(crops_with_tracks)
    n_gallery = len(gallery)
    
    # Build cost matrix
    cost_matrix = np.zeros((n_detections, n_gallery))
    
    for i, (crop, track_id) in enumerate(crops_with_tracks):
        features = reid.extract(crop)
        for j, (gid, gallery_feat) in enumerate(gallery.items()):
            similarity = 1 - cosine(features, gallery_feat)
            cost_matrix[i, j] = 1 - similarity  # Cost = 1 - sim
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build assignments
    assignments = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 0.35:  # threshold
            assignments[crops_with_tracks[i][1]] = gallery_ids[j]
        elif is_master:
            # Create new ID
            assignments[crops_with_tracks[i][1]] = create_new_id()
    
    return assignments
```

### 5.5 Thuật Toán Phân Loại Tư Thế

```python
def classify_posture(keypoints):
    """
    Phân loại tư thế từ 17 keypoints.
    
    Keypoints (COCO format):
    0: nose, 5-6: shoulders, 11-12: hips, 15-16: ankles
    
    Rules:
    - Standing: shoulders trên hips, hips trên ankles
    - Sitting: shoulders gần hips theo chiều dọc
    - Lying: shoulders và hips ngang nhau
    """
    
    # Extract key points
    nose = keypoints[0]
    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]
    l_ankle, r_ankle = keypoints[15], keypoints[16]
    
    # Calculate centers
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    ankle_y = (l_ankle[1] + r_ankle[1]) / 2
    
    # Vertical distance ratios
    torso_height = hip_y - shoulder_y
    body_height = ankle_y - shoulder_y
    
    # Horizontal spread (for lying detection)
    shoulder_spread = abs(l_shoulder[0] - r_shoulder[0])
    hip_spread = abs(l_hip[0] - r_hip[0])
    
    # Decision rules
    if body_height < 50:  # Too small to classify
        return "unknown"
    
    vertical_ratio = torso_height / max(body_height, 1)
    horizontal_ratio = max(shoulder_spread, hip_spread) / max(body_height, 1)
    
    if horizontal_ratio > 0.7:  # Body is horizontal
        return "lying"
    elif vertical_ratio > 0.6:  # Torso compressed
        return "sitting"
    else:
        return "standing"
```

---

## 6. Luồng Dữ Liệu

### 6.1 Sơ Đồ Luồng Dữ Liệu Chi Tiết

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT LAYER                                                             │
│  ═══════════                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                        │
│  │ Video    │     │ RTSP     │     │ Webcam   │                        │
│  │ Files    │     │ Stream   │     │          │                        │
│  └────┬─────┘     └────┬─────┘     └────┬─────┘                        │
│       │                │                │                               │
│       └────────────────┴────────────────┘                               │
│                        │                                                 │
│                        ▼                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │           VideoCapture                   │                            │
│  │    frame: np.ndarray (H, W, 3)          │                            │
│  └────────────────────┬────────────────────┘                            │
│                       │                                                  │
│  DETECTION LAYER      ▼                                                  │
│  ═══════════════      │                                                  │
│  ┌─────────────────────────────────────────┐                            │
│  │     YOLO Detection + ByteTrack          │                            │
│  │                                          │                            │
│  │  Input: frame (640x480x3)               │                            │
│  │  Output: List[Detection]                 │                            │
│  │    - bbox: [x1, y1, x2, y2]             │                            │
│  │    - conf: float (0-1)                   │                            │
│  │    - track_id: int                       │                            │
│  │    - keypoints: (17, 3)                  │                            │
│  └────────────────────┬────────────────────┘                            │
│                       │                                                  │
│  FEATURE LAYER        ▼                                                  │
│  ═════════════        │                                                  │
│  ┌─────────────────────────────────────────┐                            │
│  │       Feature Extraction (ReID)          │                            │
│  │                                          │                            │
│  │  For each detection:                     │                            │
│  │    crop = frame[y1:y2, x1:x2]           │                            │
│  │    embedding = extract(crop)             │                            │
│  │      → 176-dim normalized vector        │                            │
│  │                                          │                            │
│  │  Data: Dict[track_id, embedding]         │                            │
│  └────────────────────┬────────────────────┘                            │
│                       │                                                  │
│  MATCHING LAYER       ▼                                                  │
│  ══════════════       │                                                  │
│  ┌─────────────────────────────────────────┐                            │
│  │       GlobalIDManager                    │                            │
│  │                                          │                            │
│  │  ┌─────────────────────────────────┐    │                            │
│  │  │ VectorDB (FAISS)                │    │                            │
│  │  │ - search(embedding, k=20)       │    │                            │
│  │  │ - O(log N) complexity           │    │                            │
│  │  └─────────────────────────────────┘    │                            │
│  │                  │                       │                            │
│  │                  ▼                       │                            │
│  │  ┌─────────────────────────────────┐    │                            │
│  │  │ Two-Threshold Decision          │    │                            │
│  │  │ - strong: 0.65 → MATCH          │    │                            │
│  │  │ - weak: 0.40 → NEW ID           │    │                            │
│  │  │ - middle: VOTING                │    │                            │
│  │  └─────────────────────────────────┘    │                            │
│  │                  │                       │                            │
│  │                  ▼                       │                            │
│  │  ┌─────────────────────────────────┐    │                            │
│  │  │ Temporal Voting (15 frames)     │    │                            │
│  │  │ - Confirm matches over time     │    │                            │
│  │  │ - Prevent ID flicker            │    │                            │
│  │  └─────────────────────────────────┘    │                            │
│  │                                          │                            │
│  │  Output: global_id, state, score         │                            │
│  └────────────────────┬────────────────────┘                            │
│                       │                                                  │
│  ANALYSIS LAYER       ▼                                                  │
│  ══════════════       │                                                  │
│  ┌─────────────────────────────────────────┐                            │
│  │        ADL Engine (FSM)                  │                            │
│  │                                          │                            │
│  │  States per person:                      │                            │
│  │    NORMAL → POTENTIAL_FALL → FALLEN     │                            │
│  │         ↑                    ↓          │                            │
│  │         └────── RECOVERING ←─┘          │                            │
│  │                                          │                            │
│  │  Events: fall_detected, bed_exit, etc.   │                            │
│  └────────────────────┬────────────────────┘                            │
│                       │                                                  │
│  OUTPUT LAYER         ▼                                                  │
│  ════════════         │                                                  │
│  ┌─────────────────────────────────────────┐                            │
│  │          Output Generation               │                            │
│  │                                          │                            │
│  │  Video: Annotated frames (bbox, skeleton)│                            │
│  │  CSV: Event log (timestamp, event, id)   │                            │
│  │  WebSocket: Real-time JSON stream        │                            │
│  │  Database: Persistent state              │                            │
│  └─────────────────────────────────────────┘                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 State Machine cho ADL

```
┌─────────────────────────────────────────────────────────────┐
│                    ADL State Machine                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                      ┌──────────┐                           │
│           ┌─────────►│  NORMAL  │◄─────────┐               │
│           │          └────┬─────┘          │               │
│           │               │                │               │
│           │     posture == "lying"          │               │
│           │     && duration < 2s           │               │
│           │               ▼                │               │
│           │     ┌─────────────────┐        │               │
│           │     │ POTENTIAL_FALL  │        │               │
│    recover│     └────────┬────────┘        │ posture       │
│   standing│              │                 │ != "lying"    │
│           │    duration >= 2s              │               │
│           │              ▼                 │               │
│           │     ┌─────────────────┐        │               │
│           │     │     FALLEN      │────────┤               │
│           │     │  (ALERT SENT!)  │        │               │
│           │     └────────┬────────┘        │               │
│           │              │                 │               │
│           │    movement detected           │               │
│           │              ▼                 │               │
│           │     ┌─────────────────┐        │               │
│           └─────│   RECOVERING    │────────┘               │
│                 └─────────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Module Chi Tiết

### 7.1 Cấu Trúc Thư Mục

```
D:\HAVEN\
├── backend/
│   ├── core/
│   │   └── global_id_manager.py    # GlobalID orchestration
│   ├── multi/
│   │   ├── run.py                  # SequentialRunner (main)
│   │   ├── reid.py                 # EnhancedReID + DB
│   │   ├── adl.py                  # ADL Engine (FSM)
│   │   ├── visualize.py            # Drawing utilities
│   │   └── config.yaml             # Configuration
│   ├── storage/
│   │   ├── persistence.py          # SQLite + Memmap
│   │   └── vector_db.py            # FAISS wrapper
│   ├── models/
│   │   └── *.pt                    # YOLO weights
│   └── data/
│       └── multi-camera/           # Input videos
│
├── configs/
│   └── multicam.yaml               # Master config
│
├── frontend/
│   └── (WebSocket UI)
│
└── run_haven.bat                   # Entry point
```

### 7.2 Module Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                   MODULE DEPENDENCIES                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  run.py (SequentialRunner)                              │
│    ├── ultralytics.YOLO                                 │
│    ├── storage.persistence.PersistenceManager           │
│    ├── storage.vector_db.VectorDatabase                 │
│    ├── core.global_id_manager.GlobalIDManager           │
│    ├── multi.reid.MasterSlaveReIDDB                     │
│    ├── multi.adl.TrackState, classify_posture           │
│    └── multi.visualize.*                                │
│                                                          │
│  global_id_manager.py                                    │
│    ├── storage.persistence.PersistenceManager           │
│    ├── storage.vector_db.VectorDatabase, HybridMatcher  │
│    └── numpy, logging, collections                      │
│                                                          │
│  reid.py                                                 │
│    ├── cv2 (OpenCV)                                     │
│    ├── numpy                                            │
│    ├── scipy.spatial.distance.cosine                    │
│    └── scipy.optimize.linear_sum_assignment             │
│                                                          │
│  persistence.py                                          │
│    ├── sqlite3                                          │
│    ├── numpy (memmap)                                   │
│    └── threading (locks)                                │
│                                                          │
│  vector_db.py                                            │
│    ├── faiss (optional)                                 │
│    └── numpy                                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Phân Tích Độ Phức Tạp

### 8.1 Time Complexity

| Operation | Complexity | Giải thích |
|-----------|------------|------------|
| YOLO Detection | O(HW) | Convolutional, H×W pixels |
| ByteTrack | O(N×M) | N detections × M tracks |
| Feature Extract | O(HW) | HSV convert + histogram |
| Vector Search (FAISS) | O(log N) | HNSW index |
| Hungarian Assignment | O(N³) | N persons |
| Temporal Voting | O(1) | Dict lookup |
| **Per Frame Total** | **O(HW + N³)** | Dominated by detection |

### 8.2 Space Complexity

| Component | Space | Giải thích |
|-----------|-------|------------|
| Frame buffer | O(HW×3) | RGB image |
| Embeddings (memmap) | O(N×176) | N persons × 176 dims |
| FAISS index | O(N×176) | Same as embeddings |
| Vote history | O(T×G) | T tracks × G global IDs |
| SQLite DB | O(N×M) | N IDs × M observations |
| **Total Runtime** | **O(HW + N×D)** | D = embedding dim |

### 8.3 Bottleneck Analysis

```
┌──────────────────────────────────────────────────────────┐
│                 PERFORMANCE BOTTLENECKS                   │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. YOLO Inference (~70% time)                           │
│     ├── GPU: 10-15ms                                     │
│     └── CPU: 30-50ms                                     │
│     → Mitigation: Use smaller model (yolov8n)            │
│                                                           │
│  2. Feature Extraction (~15% time)                       │
│     ├── HSV conversion: 2ms                              │
│     └── Histogram: 3ms per person                        │
│     → Mitigation: Crop resize, batch processing          │
│                                                           │
│  3. Vector Search (~5% time)                             │
│     ├── N < 100: 0.1ms                                   │
│     └── N > 1000: 1-2ms                                  │
│     → Mitigation: FAISS HNSW index                       │
│                                                           │
│  4. I/O (~10% time)                                      │
│     ├── Video read: 2-5ms                                │
│     └── Video write: 5-10ms                              │
│     → Mitigation: Async I/O, buffer                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 9. Kết Luận

### 9.1 Điểm Mạnh
- ✅ **Kiến trúc Master-Slave** rõ ràng, dễ mở rộng
- ✅ **Two-threshold decision** giảm false matches
- ✅ **Temporal voting** chống ID flicker
- ✅ **FAISS index** cho O(log N) search
- ✅ **Persistent storage** qua restarts

### 9.2 Điểm Cần Cải Thiện
- ⚠️ **Color histogram** nhạy với thay đổi ánh sáng
- ⚠️ **Single embedding** không robust với thay quần áo
- ⚠️ **Sequential processing** chưa tối ưu cho realtime

### 9.3 Đề Xuất Nâng Cấp
1. **Deep ReID** (OSNet/ResNet50) thay HSV histogram
2. **Multi-prototype memory** cho clothing change
3. **Parallel camera processing** tăng throughput
4. **Face embedding** làm primary signal

---

**Tài liệu viết bởi**: Chuyên gia DSA/OOP  
**Ngày**: 2026-02-03  
**Phiên bản**: 2.0  
**Dự án**: HAVEN Multi-Camera Person Re-Identification
