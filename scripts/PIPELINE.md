# CPose — Phân Tích Pipeline Thuật Toán & Lỗ Hổng Kỹ Thuật

> **Tài liệu phân tích kỹ thuật** cho hệ thống CPose: Pose Estimation · ADL Classification · Person Re-ID xuyên camera.  
> Dựa trên toàn bộ source code và tài liệu thiết kế trong repo.

---

## 1. Tổng Quan Kiến Trúc Pipeline

CPose gồm 3 phase xử lý tuần tự + 1 lớp nhận dạng danh tính người (ReID) hoạt động xuyên camera.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CPose Algorithm Pipeline                      │
│                                                                     │
│  RTSP Streams (N cameras)                                           │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────┐                                               │
│  │  PHASE 1         │  YOLOv8n → Person detect → FSM → MP4 clip     │
│  │  recorder │  (IDLE → ARMED → RECORDING → POST_ROLL)       │
│  └────────┬─────────┘                                               │
│           │ raw_videos/*.mp4                                        │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │  PHASE 2         │  YOLOv11n → BBox → PNG frame + labels.txt     │
│  │  analyzer │  (offline batch processing)                   │
│  └────────┬─────────┘                                               │
│           │ output_labels/<stem>/                                   │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3 — Pose + ADL                                        │   │
│  │  recognizer + pose_utils                              │   │
│  │                                                              │   │
│  │  YOLOv11n-pose → 17 COCO Keypoints                           │   │
│  │       ↓                                                      │   │
│  │  Sliding Window (W=30 frames)                                │   │
│  │       ↓                                                      │   │
│  │  Rule-Based ADL Classifier (priority decision tree)          │   │
│  │       ↓                                                      │   │
│  │  ADL Labels: standing/sitting/walking/lying_down/            │   │
│  │              falling/reaching/bending/unknown                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  REALTIME Re-ID LAYER (HavenPose Design — chưa hoàn chỉnh)   │   │
│  │                                                              │   │
│  │  DeepSort Tracker → track_id                                 │   │
│  │       ↓                                                      │   │
│  │  EnhancedReID (HSV Histogram + Hu Moments)                   │   │
│  │       ↓                                                      │   │
│  │  MasterSlaveReIDDB + Temporal Voting (confirm_frames=3)      │   │
│  │       ↓                                                      │   │
│  │  GlobalIDManager (Master/Slave hierarchy)                    │   │
│  │       ↓                                                      │   │
│  │  FAISS VectorDB (cosine ANN search)                          │   │
│  │       ↓                                                      │   │
│  │  PersistenceManager (SQLite + numpy memmap)                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi Tiết Từng Module Thuật Toán

### 2.1 Phase 1 — Person-Triggered Recording

**File:** `app/services/recorder.py`

**Thuật toán FSM (Finite State Machine):**

```
State: IDLE → ARMED → RECORDING → POST_ROLL → IDLE
         ↑                                       │
         └───────────────────────────────────────┘

Trigger:
  IDLE    → ARMED     : person detected (YOLOv8n, conf > 0.65)
  ARMED   → RECORDING : N=3 consecutive frames xác nhận có người
  ARMED   → IDLE      : person disappears before confirmation
  RECORDING→POST_ROLL : person gone (no_person_cnt++)
  POST_ROLL→RECORDING : person returns
  POST_ROLL→IDLE      : no_person_cnt >= post_frames (≈5s×fps)
```

**Hyperparameters:**

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `PERSON_CONF_THRESHOLD` | 0.65 | Confidence tối thiểu để chấp nhận detection |
| `TRIGGER_MIN_CONSECUTIVE` | 3 | Số frame liên tiếp phải có người để ARMED→RECORDING |
| `PRE_ROLL_SECONDS` | 3 | Buffer trước sự kiện (deque circular) |
| `POST_ROLL_SECONDS` | 5 | Buffer sau khi mất detection |
| `MIN_BOX_AREA_RATIO` | 0.0015 | Lọc bbox quá nhỏ (tránh false positive xa) |
| `INFERENCE_EVERY` | 3 | Chạy YOLO mỗi N frame (giảm tải CPU) |

**Flow kỹ thuật:**
```python
pre_buffer = deque(maxlen=pre_buf_size)   # circular, ~3s × fps frames
↓
cap.read() → frame
↓ (mỗi INFERENCE_EVERY frame)
model.predict(frame, classes=[0], conf=threshold)
  → filter by box area ratio
  → best_conf > 0 → person_now = True
↓
FSM transition + VideoWriter flush pre_buffer on RECORDING start
```

---

### 2.2 Phase 2 — Offline Bounding Box Labeling

**File:** `app/services/analyzer.py`

**Thuật toán:** YOLOv11n inference đơn giản, không tracking.

```
for each frame:
  results = YOLOv11n(frame, classes=[0], conf=0.50)
  if detections:
    save PNG frame
    write (frame_id, x1, y1, x2, y2) to labels.txt
```

**Không có:** tracking, identity, temporal smoothing. Đây là offline annotation tool thuần túy.

---

### 2.3 Phase 3 — Pose Estimation + ADL Classification

**Files:** `app/services/recognizer.py`, `app/utils/pose_utils.py`

#### 2.3.1 Pose Estimation

**Model:** YOLOv11n-pose → 17 keypoints COCO format (x, y, confidence)

```
COCO 17 Keypoints:
[0]nose  [1]L_eye  [2]R_eye  [3]L_ear  [4]R_ear
[5]L_shoulder  [6]R_shoulder  [7]L_elbow  [8]R_elbow
[9]L_wrist  [10]R_wrist  [11]L_hip  [12]R_hip
[13]L_knee  [14]R_knee  [15]L_ankle  [16]R_ankle
```

#### 2.3.2 ADL Sliding Window Classifier

**Cơ chế:**
- Mỗi person được gán `person_id` = index trong frame (0, 1, 2...)
- Buffer: `deque(maxlen=30)` → khi đủ 30 frames mới classify
- Gọi `rule_based_adl(window, config)` → `(label, confidence)`

#### 2.3.3 Rule-Based ADL Classifier (pose_utils.py)

**Pipeline tính toán:**

```
Step 1: Lấy keypoints đủ confidence (>= 0.3)
Step 2: Tính shoulder_mid, hip_mid
Step 3: torso_angle = atan2(|Δx|, |Δy|)  → góc nghiêng thân
Step 4: avg_knee_angle = mean(calc_angle(hip, knee, ankle))
Step 5: aspect_ratio = bbox_width / bbox_height
Step 6: walk_velocity = calc_velocity(ankle_positions qua W frames)
Step 7: wrists_above_shoulders (boolean)

Priority Decision Tree:
  IF torso_angle > 68° AND velocity > 8.0×1.1  → "falling"   (conf=0.88)
  IF torso_angle > 68° AND aspect_ratio > 1.15 → "lying_down" (conf=0.84)
  IF knee_angle < 150° AND velocity < 8.0       → "sitting"   (conf=0.82)
  IF torso_angle > 45° AND velocity < 8.0×0.6   → "bending"   (conf=0.78)
  IF any wrist above shoulder                    → "reaching"  (conf=0.76)
  IF velocity > 8.0                              → "walking"   (conf=0.79)
  ELSE                                           → "standing"  (conf=0.75)
  IF visible_keypoints < 8                       → "unknown"   (conf=0.20)
```

**Hàm hình học:**
```python
calc_angle(p1, vertex, p2):
    a = p1 - vertex; b = p2 - vertex
    cos = dot(a,b) / (|a|×|b|)
    return degrees(acos(clip(cos,-1,1)))

calc_velocity(ankle_positions):
    displacements = [|pos[i+1] - pos[i]| for i in range(len-1)]
    return mean(displacements)
```

---

### 2.4 ReID Pipeline — Nhận Dạng Người Xuyên Camera

#### 2.4.1 EnhancedReID Feature Extractor (app/core/reid.py)

**Đặc trưng trích xuất từ person crop:**

```
Body crop split thành 3 phần:
  Head  : 0%   → 20% chiều cao (HSV histogram)
  Body  : 20%  → 70% chiều cao (HSV histogram - trọng số cao nhất)
  Legs  : 70%  → 100% chiều cao (HSV histogram)

Mỗi phần:
  H channel: 24 bins [0-180]   → màu sắc chính
  S channel: 16 bins [0-256]   → độ bão hòa
  V channel: 16 bins [0-256]   → độ sáng (quan trọng cho cross-camera)
  → 56 features × 3 parts = 168 color features

Shape features:
  Hu Moments (7 giá trị) từ grayscale
  → log transform: -sign(h)×log10(|h|+ε)
  → normalize by L2 norm

+ aspect_ratio (1 feature)

Total: 168 + 7 + 1 = 176-dimensional feature vector
```

**Similarity metric:** Cosine distance (1 - cosine similarity) via `scipy.spatial.distance.cosine`

**Top-K averaging:**
```python
sims = [cosine_sim(query, stored) for stored in person_features]
top_k = sorted(sims, reverse=True)[:5]
final_score = mean(top_k)
```

#### 2.4.2 MasterSlaveReIDDB — Temporal Voting

```
Master Camera (cam01):
  Person crop → EnhancedReID.extract() → features
  → _find_best_match() (linear scan all persons)
  → IF sim >= 0.65 (strong): temporal_vote[track_id][gid] += 1
                               IF votes >= 3: CONFIRMED → return GlobalID
  → IF no match: temporal_vote[track_id][None] += 1
                 IF votes >= 3: create new GlobalID

Slave Camera (cam02/03/04):
  Person crop → features
  → _find_best_match() (only match, never create)
  → IF sim >= 0.65: temporal_vote → CONFIRMED
  → ELSE: return None (UNKNOWN)
```

**Hungarian Algorithm (multi-person):**
```python
cost_matrix[i][j] = 1 - cosine_sim(track_i_features, person_j_features)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
→ optimal global assignment minimizing total cost
```

#### 2.4.3 GlobalIDManager (app/core/global_id.py)

```
Hierarchy:
  MASTER cam → _assign_master():
    FAISS search → strong match (>=0.65) → temporal voting → CONFIRMED
                 → no strong match → register_global_id() (create new)

  SLAVE cam  → _assign_slave():
    _get_valid_candidates() → FAISS search (allowed_ids only)
    → strong match (>=0.65) → temporal voting → CONFIRMED
    → weak match (0.45-0.65) → UNK
    → no match (<0.45) → UNK

  UNK handling:
    → IoU resurrection: compare bbox với last_unk_bboxes
    → IF IoU > 0.3: resurrect same UNK ID
    → ELSE: create new UNKn
```

#### 2.4.4 FAISS Vector Database (app/storage/vector_db.py)

```
Auto-upgrade strategy (metric: cosine via inner product):
  N < 1,000   : IndexFlatIP (exact search, O(N))
  1,000-10,000: IndexHNSWFlat (M=32, ef_construction=200)
  N >= 10,000 : IndexIVFFlat (nlist=100, nprobe=10)

Search: top-k=20, L2-normalize trước khi add
```

#### 2.4.5 Persistence (app/storage/persistence.py)

```
SQLite (haven_state.db):
  - global_ids: (global_id, created_at, first_camera, embedding_idx, is_active)
  - camera_appearances: log mỗi lần nhìn thấy
  - person_profiles: (name, age, gender)
  - metadata: next_global_id counter

numpy memmap (embeddings.npy):
  - Shape: (N, 512) float32 — raw binary (không phải .npy format!)
  - EMA update: emb_new = α×emb_new + (1-α)×emb_old, α=0.3
  - Auto-expand: khi embedding_idx >= capacity, +1000 rows
```

---

### 2.5 Tracking — DeepSort

**File:** `app/core/tracking.py`

```python
DeepSort(
  max_age=30,            # frame tối đa không thấy trước khi xóa track
  n_init=3,              # frame tối thiểu để khởi tạo track mới
  max_iou_distance=0.7,  # IoU matching threshold
  max_cosine_distance=0.4, # appearance distance threshold
  embedder='mobilenet',  # deep appearance embedder
  half=True              # FP16 inference
)
```

---

## 3. Tương Quan Với Các Paper Tham Khảo

| Paper | Thuật Toán | Trạng Thái Trong Code |
|-------|-----------|----------------------|
| **ST-GCN** (AAAI 2018) | Graph CNN trên skeleton graph temporal | ❌ Không triển khai — code dùng rule-based |
| **SkateFormer** | Skeleton Transformer với temporal attention | ❌ Không triển khai |
| **BlockGCN** | Topology-aware graph convolution | ❌ Không triển khai |
| **ChannelWise Topology** | Per-channel dynamic graph | ❌ Không triển khai |
| **Autoregressive Hypergraph** | Adaptive hypergraph neural net | ❌ Không triển khai |
| **TSM** | Temporal Shift Module cho video CNN | ❌ Không triển khai |
| **RTMPose / RTMO** | Real-time multi-person pose | ✅ Thay thế bằng YOLOv11n-pose |
| **From Poses to Identity** | Pose-based person ReID | ⚠️ Chỉ dùng color histogram (EnhancedReID) |
| **Smarthome ADL** | ADL benchmark dataset + evaluation | ⚠️ ADL classes tương đồng nhưng metric khác |
| **Enhancing Action Recognition** | Leveraging pose for action | ❌ Không tích hợp |

**Kết luận:** Code triển khai thuật toán ADL ở mức **rule-based heuristic**, trong khi các paper tham khảo đề xuất **deep learning graph-based** approach. Đây là khoảng cách lớn nhất giữa tài liệu nghiên cứu và implementation thực tế.

---

## 4. Lỗ Hổng Kỹ Thuật (Bugs & Design Flaws)

### 🔴 CRITICAL — Gây Sai Kết Quả Nghiêm Trọng

---

#### Bug #1: Hai ADL Classifier Song Song, Mâu Thuẫn Nhau

**Vị trí:** `app/utils/pose_utils.py` vs `app/core/adl.py`

**Vấn đề:**
Hệ thống có **2 bộ classifier ADL độc lập** với threshold khác nhau hoàn toàn:

| Thông số | `pose_utils.py` | `adl.py` |
|---------|----------------|---------|
| Torso angle lying | `FALLING_TORSO_ANGLE = 68°` | `TORSO_ANGLE_LAYING = 35°` |
| Aspect ratio lying | `LYING_ASPECT_RATIO = 1.15` | `ASPECT_RATIO_LAYING = 0.9` |
| Knee angle sitting | `knee_bend_angle = 150°` | `KNEE_ANGLE_SITTING = 140°` |
| ADL labels | `lying_down, falling, bending...` | `FALL_DOWN, SITTING, WALKING` |
| Confidence | Hardcoded 0.75–0.88 | Không có confidence |

`phase3_recognizer.py` gọi `pose_utils.rule_based_adl()`.  
`adl.py` + `adl.py.classify_posture()` được dùng trong realtime pipeline (`core/`).

**Hậu quả:** Cùng một pose sẽ ra hai nhãn ADL khác nhau tùy component nào được gọi. Kết quả không thể so sánh hoặc tích hợp.

**Sửa:** Thống nhất 1 classifier duy nhất. Xóa `adl.py` hoặc refactor vào `PoseADLEngine` chung.

---

#### Bug #2: `person_id` Trong Phase 3 Không Ổn Định

**Vị trí:** `app/services/phase3_recognizer.py`, dòng trong `_process_clip()`

```python
for person_id, (person_xy, person_conf) in enumerate(zip(people_xy, people_conf)):
    person_windows[person_id].append(...)
```

**Vấn đề:** `person_id` = vị trí index trong danh sách detection của frame hiện tại (0, 1, 2...), **không phải track ID**. Khi 1 người ra/vào frame hoặc thứ tự detection thay đổi, `person_id=0` của frame 50 có thể là người khác hoàn toàn so với `person_id=0` của frame 49.

**Hậu quả:** Sliding window `person_windows[0]` chứa keypoints hỗn hợp của nhiều người khác nhau → ADL classification sai hoàn toàn.

**Sửa:** Tích hợp DeepSort tracker để gán `stable_track_id` thay vì `enumerate`.

---

#### Bug #3: Vector Index Không Được Cập Nhật Sau EMA Update

**Vị trí:** `app/core/global_id.py`, hàm `_update_vector_index()`

```python
def _update_vector_index(self, global_id: int, new_embedding: np.ndarray):
    # For now, simple approach: index will be periodically rebuilt
    pass  # ← EMPTY! Không làm gì cả
```

**Vấn đề:** Khi `PersistenceManager.update_appearance()` EMA-update embedding trong memmap, FAISS index **không được cập nhật**. Search tiếp tục dùng embedding cũ từ lần đầu đăng ký.

**Hậu quả:** Sau nhiều lần cập nhật, embedding trong memmap và FAISS diverge ngày càng xa. Matching quality giảm dần theo thời gian.

**Sửa:** Implement `_update_vector_index()` bằng cách rebuild index định kỳ hoặc dùng `faiss.write_index` + `IndexIDMap` để update individual vectors.

---

#### Bug #4: `embeddings.npy` Là Raw Binary, Không Phải NPY Format

**Vị trí:** `app/storage/persistence.py`, `_init_embeddings()`

```python
self.embeddings = np.memmap(
    str(self.embeddings_path),  # tên file: embeddings.npy
    dtype='float32',
    mode='w+',
    shape=(initial_size, self.embedding_dim)
)
```

**Vấn đề:** `np.memmap` tạo raw binary array (không có `.npy` header). Nhưng file được đặt tên `.npy`. Nếu ai đó gọi `np.load("embeddings.npy")`, sẽ nhận được kết quả sai hoặc exception.

**Logic phục hồi cũng sai:**
```python
file_size = self.embeddings_path.stat().st_size
n_rows = file_size // row_size  # ← Tính đúng cho raw binary
# nhưng nếu file là .npy thật thì sẽ sai vì có header 128 bytes
```

**Sửa:** Đổi tên file thành `embeddings.bin` hoặc thêm comment rõ ràng + validate.

---

### 🟠 HIGH — Ảnh Hưởng Hiệu Năng và Độ Tin Cậy

---

#### Bug #5: Memory Leak trong `pending_ids` và `track_to_global`

**Vị trí:** `app/core/reid.py`, `MasterSlaveReIDDB`

```python
self.pending_ids = {}     # track_id → {votes, best_match, features}
self.track_to_global = {} # track_id → global_id (confirmed)
```

**Vấn đề:** Khi một track bị DeepSort xóa (sau `max_age=30` frames), entry trong `pending_ids` và `track_to_global` **không bao giờ được dọn**.

**Hậu quả:** Trong session dài (ghi hình 8 giờ với nhiều camera), 2 dict này tích lũy hàng nghìn stale entries → memory leak.

**Sửa:** Hook vào DeepSort's `deleted_tracks` callback hoặc implement LRU eviction.

---

#### Bug #6: `_expand_embeddings()` Dùng `shutil.move` Khi File Đang Mở

**Vị trí:** `app/storage/persistence.py`, `_expand_embeddings()`

```python
del self.embeddings              # giải phóng memmap reference
shutil.move(str(new_path), str(self.embeddings_path))  # replace file
```

**Vấn đề:** Trên **Windows**, `np.memmap` giữ file handle ngay cả sau `del`. Python's garbage collector không đảm bảo file được close ngay lập tức. `shutil.move` sẽ raise `PermissionError`.

**Hậu quả:** Crash khi số GlobalID vượt 10,000 (initial capacity).

**Sửa:**
```python
del self.embeddings
import gc; gc.collect()  # force release trên Windows
shutil.move(...)
```

---

#### Bug #7: SocketIO Emit Từ Non-Main Thread Không An Toàn Trong Eventlet Mode

**Vị trí:** `app/utils/file_handler.py`, `enforce_storage_limit()`

```python
from app import socketio
socketio.emit("storage_warning", {...})  # gọi trong CameraThread
```

**Vấn đề:** `CameraThread` là `threading.Thread` thông thường, không phải eventlet green thread. Gọi `socketio.emit` từ đây trong `eventlet` async mode có thể gây **race condition** hoặc deadlock.

**Sửa:** Dùng `socketio.emit(..., namespace='/', to=None)` hoặc wrap bằng `eventlet.spawn_after(0, emit_fn)`.

---

#### Bug #8: EnhancedReID Nhạy Với Thay Đổi Ánh Sáng Giữa Các Camera

**Vị trí:** `app/core/reid.py`, `EnhancedReID.extract()`

**Vấn đề:** Feature vector gồm H (hue), S (saturation), V (value) channel histograms. Trong cross-camera ReID, ánh sáng của từng camera thường khác nhau (indoor/outdoor, color temperature, exposure). **V channel** (brightness) sẽ shift đáng kể giữa các camera → cosine similarity giảm mạnh mặc dù cùng người.

**Không có:** Camera normalization, illumination correction, whitening.

**Hậu quả:** Matching accuracy thấp khi camera có điều kiện ánh sáng khác nhau (thường xảy ra trong thực tế phòng lab).

**Sửa tốt:** Dùng histogram equalization trên V channel, hoặc thêm camera-specific color calibration matrix.

---

#### Bug #9: Phase 3 Lưu Overlay PNG Cho Mọi Frame Có Người

**Vị trí:** `app/services/phase3_recognizer.py`

```python
if overlay_frame is not None and persons_detected > 0:
    overlay_path = clip_output_dir / f"{clip.stem}_overlay_{frame_id:04d}.png"
    cv2.imwrite(str(overlay_path), overlay_frame)  # mỗi frame!
```

**Vấn đề:** Nếu clip 10 phút, 30fps, có người liên tục → **18,000 PNG files** (~1-2 GB). Không có sampling, không có giới hạn.

**Hậu quả:** Disk fill-up nhanh chóng. File system có thể bị chậm với hàng chục nghìn files trong 1 folder.

**Sửa:** Save mỗi N frame (ví dụ `if frame_id % 30 == 0`) hoặc chỉ save keyframes.

---

### 🟡 MEDIUM — Thiếu Sót Thiết Kế

---

#### Bug #10: `_find_best_match()` Là O(N×M) — Không Dùng FAISS Đã Xây Dựng

**Vị trí:** `app/core/reid.py`, `_find_best_match()`

```python
for gid in self.persons:                    # O(N) persons
    for stored_feat in info['features']:    # O(M) features each
        sim = 1 - cosine(features, stored_feat)  # O(D) dimensions
```

**Vấn đề:** `MasterSlaveReIDDB` tự làm linear scan riêng, **bỏ qua hoàn toàn** `VectorDatabase` đã được xây dựng trong `app/storage/vector_db.py` với FAISS ANN search.

**Hai hệ thống ReID song song, không kết nối:**
- `app/core/reid.py` → color histogram, linear scan → dùng trong realtime demo scripts
- `app/core/global_id.py` + `app/storage/vector_db.py` → FAISS, proper embedding → architecture đúng nhưng integration chưa hoàn tất

---

#### Bug #11: `_get_valid_candidates()` Luôn Trả Về Tất Cả GlobalID

**Vị trí:** `app/core/global_id.py`

```python
def _get_valid_candidates(self, camera: str, frame_time: float) -> List[int]:
    # Get all active GlobalIDs
    _, global_ids = self.persistence.get_all_embeddings(active_only=True)
    return global_ids.tolist()  # TẤT CẢ, không lọc
    # TODO: Implement camera graph transition time filtering.
```

**Vấn đề:** Chú thích nói sẽ dùng spatiotemporal gating (dựa trên thời gian di chuyển giữa camera), nhưng thực tế trả về tất cả GlobalID. Một người vừa được nhìn thấy ở camera A (phòng 1) sẽ vẫn là candidate ở camera B (phòng khác) ngay lập tức.

**Hậu quả:** False positive matching — người ở camera B bị nhận nhầm là người vừa seen ở camera A dù về mặt vật lý không thể di chuyển ngay.

---

#### Bug #12: `hungarian_assign()` Có Thể Gọi `match_only()` Với `track_id=None` Nhầm

**Vị trí:** `app/core/reid.py`, `hungarian_assign()`

```python
result[track_id] = self.match_only(crop, cam_id, track_id)
```

Nhưng với slave camera theo `is_master=False`, sau Hungarian assignment:
```python
elif is_master:
    result[track_id] = self.register_new(crop, cam_id, track_id)
else:
    result[track_id] = None
```

Track_ids trả về `None` từ `register_new` khi pending (chưa đủ votes) sẽ bị ghi đè thành `None` trong result, làm mất thông tin pending.

---

#### Bug #13: Không Có Camera Role Enforcement Trong Code

**Vị trí:** `app/core/global_id.py`

Thiết kế yêu cầu:
- `cam01` = MASTER (có thể tạo GlobalID mới)
- `cam02/03/04` = SLAVE (chỉ match)
- `cam04` = STRICT (alert nếu UNKNOWN)

Nhưng trong code, role được check bằng:
```python
if camera == self.master_camera:  # string comparison
    return self._assign_master(...)
elif camera in self.slave_cameras:
    return self._assign_slave(...)
```

Không có `STRICT` camera logic nào được implement. Không có security alert, không có intruder detection. Đây là tính năng còn trên giấy.

---

### 🟢 LOW — Code Quality Issues

---

#### Issue #14: Config Fragmentation — 3 Nơi Định Nghĩa Cùng Threshold

ADL thresholds được định nghĩa tại:
1. `configs/config.yaml` → section `pose_utils`, `adl`, `phase3`
2. `app/utils/pose_utils.py` → hardcoded defaults trong module-level vars
3. `app/core/adl.py` → `ADLConfig` class với defaults khác

Khi `config.yaml` thay đổi, chỉ có `pose_utils.py` đọc đúng qua `get_runtime_section`. `adl.py` cũng đọc config nhưng section name khác (`adl` vs `pose_utils`).

---

#### Issue #15: `POSE_CONFIG_FILE = UNIFIED_CONFIG_FILE` — Hai Tên Cho Một File

```python
# app/__init__.py
UNIFIED_CONFIG_FILE = CONFIGS_DIR / "config.yaml"
POSE_CONFIG_FILE = UNIFIED_CONFIG_FILE  # same file, different alias
```

`phase3_recognizer.py` nhận `config_path=_app_module.POSE_CONFIG_FILE` nhưng `CLAUDE.md` nói file là `pose_adl.yaml`. Thực tế `POSE_CONFIG_FILE` trỏ tới `config.yaml`. Nếu ai tách ra file riêng thì cần update cả alias.

---

#### Issue #16: `last_unk_bboxes` Resize Không An Toàn (Dict Pop Order)

```python
if len(self.last_unk_bboxes[camera]) > self.max_unk_per_video:
    oldest = list(self.last_unk_bboxes[camera].keys())[0]  # ← lấy key đầu tiên
    del self.last_unk_bboxes[camera][oldest]
```

Trong Python 3.7+, dict duy trì insertion order → `keys()[0]` là item cũ nhất. Nhưng logic này giả định điều đó. Nên dùng `collections.OrderedDict` hoặc comment rõ dependency on insertion order.

---

## 5. Sơ Đồ Phụ Thuộc Module

```
routes.py
  ├── phase1_recorder.py (RecorderManager)
  │   └── [YOLO YOLOv8n] ← model load khi start
  ├── phase2_analyzer.py (Analyzer)
  │   └── [YOLO YOLOv11n]
  ├── phase3_recognizer.py (PoseADLRecognizer)
  │   ├── pose_utils.py → rule_based_adl(), calc_angle(), draw_skeleton()
  │   └── [YOLO YOLOv11n-pose]
  ├── file_handler.py (StorageManager)
  └── stream_probe.py (StreamProber)

core/global_id.py (GlobalIDManager)
  ├── storage/persistence.py (PersistenceManager)
  │   └── [SQLite + numpy memmap]
  ├── storage/vector_db.py (VectorDatabase + HybridMatcher)
  │   └── [FAISS]
  └── [NOT CONNECTED TO routes.py — chưa integrated]

core/reid.py (MasterSlaveReIDDB + EnhancedReID)
  └── [NOT CONNECTED TO GlobalIDManager — parallel system]

core/tracking.py (Tracker → DeepSort)
  └── [NOT CONNECTED TO phase3_recognizer]

core/adl.py (ADLConfig + classify_posture + TrackState)
  └── [DUPLICATE of pose_utils.py — không dùng trong Phase 3]
```

**Kết luận phụ thuộc:** `GlobalIDManager`, `VectorDatabase`, `PersistenceManager`, `MasterSlaveReIDDB`, `DeepSort Tracker`, và `ADLConfig` tồn tại nhưng **chưa được kết nối vào pipeline chính** (routes.py → Phase 1/2/3).

---

## 6. So Sánh Thiết Kế vs Triển Khai Thực Tế

| Tính Năng | Theo CLAUDE.md / Pipeline.md | Thực Tế Trong Code |
|-----------|-----------------------------|--------------------|
| ADL Classification | Rule-based (phase 3) | ✅ Triển khai — 2 bản song song |
| Person tracking Phase 3 | Sliding window per person | ⚠️ person_id không stable |
| ReID xuyên camera | Color histogram + temporal voting | ✅ Có code, ❌ Không kết nối vào routes |
| FAISS vector search | Auto-upgrade Flat→HNSW→IVF | ✅ Code hoàn chỉnh, ❌ index không update |
| Master/Slave camera | Strict hierarchy, STRICT alert | ⚠️ Hierarchy OK, STRICT chưa có |
| Camera streaming | JPEG snapshot @ 6fps | ✅ Hoạt động |
| Storage management | Prune oldest, 80% target | ✅ Hoạt động |
| Realtime Live Mode | Tab 6 với ReID live | ❌ Chưa có trong index.html |
| Face recognition | YuNet + SFace | ❌ Chưa integrate (insightface optional) |
| Security/intruder alert | SocketIO security_event | ❌ Chưa implement |

---

## 7. Đề Xuất Ưu Tiên Sửa Chữa

### Ưu Tiên 1 — Sửa Ngay (Block Correctness)

1. **[Bug #2]** Gán stable track_id cho Phase 3 → tích hợp DeepSort hoặc dùng YOLO's built-in tracking (`model.track()`)
2. **[Bug #1]** Xóa `adl.py` hoặc chọn 1 classifier duy nhất, thống nhất thresholds
3. **[Bug #3]** Implement `_update_vector_index()` với periodic rebuild

### Ưu Tiên 2 — Sửa Trước Demo

4. **[Bug #9]** Sampling overlay frames (save mỗi 30 frames)
5. **[Bug #5]** Thêm cleanup cho `pending_ids` và `track_to_global` khi track expire
6. **[Bug #6]** Thêm `gc.collect()` trước `shutil.move()` trong `_expand_embeddings()`

### Ưu Tiên 3 — Cải Thiện Chất Lượng

7. **[Bug #8]** Thêm V-channel normalization hoặc histogram equalization trong EnhancedReID
8. **[Bug #10]** Kết nối `MasterSlaveReIDDB` với `VectorDatabase` để dùng FAISS thay linear scan
9. **[Bug #11]** Implement spatiotemporal gating trong `_get_valid_candidates()`

### Nâng Cấp Thuật Toán (Long Term)

10. Thay rule-based ADL bằng **ST-GCN** hoặc **SkateFormer** (các paper đã tham khảo) để tăng accuracy từ ~70% (rule-based) lên ~90%+
11. Thay color histogram ReID bằng **pose-based ReID** (theo paper "From Poses to Identity")
12. Tích hợp **TSM (Temporal Shift Module)** để xử lý temporal dependencies tốt hơn trong ADL

---

## 8. Tóm Tắt Điểm Mạnh

| Điểm Mạnh | Mô Tả |
|-----------|-------|
| **FSM Phase 1** | State machine rõ ràng, pre/post buffer đúng thiết kế |
| **FAISS VectorDB** | Code đầy đủ, auto-upgrade strategy hợp lý |
| **PersistenceManager** | SQLite + memmap đúng hướng cho production |
| **GlobalID Hierarchy** | Master/Slave separation theo đúng nguyên tắc |
| **Temporal Voting** | Giảm false positive ReID hiệu quả |
| **Config System** | `get_runtime_section()` + `lru_cache` clean |
| **Dashboard SPA** | SocketIO integration đầy đủ, responsive UI |

---
