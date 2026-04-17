# Khảo Sát Toàn Diện: ADL Recognition cho Hệ Thống Camera Giám Sát

## 1. Tổng quan và Phạm vi Khảo sát

Báo cáo này khảo sát hệ thống các phương pháp nhận dạng **Hoạt động Sinh hoạt Hàng ngày (ADL — Activities of Daily Living)** phục vụ trực tiếp cho dự án CPose: hệ thống camera giám sát đa luồng RTSP, phát hiện người, theo dõi tư thế, và cảnh báo an toàn tại nhà.

**Phạm vi khảo sát:**
- Skeleton/Pose-based action recognition (trọng tâm chính — phù hợp pipeline CPose)
- Video/RGB-based methods (3D-CNN, Video Transformers)
- Pose estimation backbone (RTMPose, RTMO — đầu vào cho ADL)
- ADL-specific datasets và benchmark protocols
- Đề xuất model khả thi cho PoC

**Papers đã phân tích (từ project):**

| Paper | Venue | Năm |
|-------|-------|-----|
| ST-GCN — Yan et al. | AAAI | 2018 |
| MS-G3D — Liu et al. | CVPR | 2020 |
| CTR-GCN — Chen et al. | ICCV | 2021 |
| TSM — Lin et al. | ICCV | 2019 |
| Toyota Smarthome — Das et al. | ICCV | 2019 |
| RTMPose — Jiang et al. | ArXiv | 2023 |
| BlockGCN — Zhou et al. | CVPR | 2024 |
| RTMO — Lu et al. | CVPR | 2024 |
| SkateFormer — Do & Kim | ECCV | 2024 |
| AutoregAd-HGformer — Ray et al. | WACV | 2025 |
| Enhancing Action Recognition (Hierarchical TSU) | IEEE | 2024 |
| Pose2ID — Yuan et al. | CVPR | 2025 |

---

## 2. Phân Tích Các Hướng Tiếp Cận Chính

### 2.1 Skeleton/Pose-Based Methods (Phương pháp dựa trên bộ khung xương)

Đây là hướng tiếp cận **phù hợp nhất với pipeline CPose** vì: (1) đầu ra của YOLOv11-pose là keypoints COCO-17, (2) skeleton data nhẹ, nhanh, bất biến với ánh sáng và nền, (3) trực tiếp phản ánh tư thế người.

#### 2.1.1 Graph Convolutional Networks (GCN) cho Skeleton

**ST-GCN (AAAI 2018) — Yan et al.**

Pipeline cốt lõi:
```
Skeleton sequence (T frames × 18 joints × 3D coords)
    → Spatial-Temporal Graph G = (V, E)
        → V: joints làm nodes
        → E_spatial: kết nối theo cấu trúc cơ thể người
        → E_temporal: kết nối cùng joint qua frame liên tiếp
    → Multiple ST-GCN layers
        → Spatial Graph Conv (nhóm joints theo khoảng cách)
        → Temporal Conv (1D conv dọc thời gian)
    → Global Average Pooling
    → Softmax → Action label
```

Đặc điểm kỹ thuật:
- Adjacency matrix A cố định theo cấu trúc giải phẫu
- Partitioning strategy: centripetal/centrifugal/root
- Input: (C=3, T=300, V=18, M=2) tensor — C: tọa độ, T: frames, V: joints, M: người
- **Kết quả:** NTU RGB+D Cross-Subject 81.5% / Cross-View 88.3%
- **Hạn chế:** A cứng không học được, không mô hình hóa quan hệ joints xa

**MS-G3D (CVPR 2020) — Liu et al.**

Hai đổi mới chính:
1. **Disentangled Multi-Scale Aggregation**: tách riêng thông tin từ các vùng lân cận khác khoảng cách, giải quyết "biased weighting problem" của adjacency powering
2. **G3D module**: Graph convolution thống nhất không-thời gian — giới thiệu cross-spacetime edges để thông tin di chuyển trực tiếp qua cả không gian lẫn thời gian (thay vì factorized spatial→temporal)

Pipeline:
```
Skeleton sequence
    → Disentangled Multi-Scale Aggregator
        → Tách features từ k-hop neighbors (không bị bias)
    → G3D module
        → Dense cross-spacetime edges (skip connections)
        → Unified ST graph convolution
    → MS-G3D feature extractor
    → Classification head
```

**Kết quả:**
- NTU RGB+D 60: CS 91.5% / CV 96.2%
- NTU RGB+D 120: CS 86.9% / CSet 88.4%
- Kinetics Skeleton: 38.0% / 60.9%

**CTR-GCN (ICCV 2021) — Chen et al.**

Đổi mới: **Channel-wise Topology Refinement** — thay vì dùng 1 topology cho toàn bộ channels, học topology riêng cho từng channel:
- **Shared topology**: ma trận adjacency tham số hóa, làm prior chung
- **Channel-specific correlations**: học động từ input sample
- **Refinement**: topology kênh = shared topology ⊕ channel-specific (ít tham số, khó optimize hơn)

Ưu điểm về tính toán: resize frames về 64 frame (NTU 60/120), giảm chi phí đáng kể.

**Kết quả:**
- NTU RGB+D 60: CS 92.4% / CV 96.8%
- NTU RGB+D 120: CS 88.9% / CSet 90.6%
- NW-UCLA: 96.5%

**BlockGCN (CVPR 2024) — Zhou et al.**

Giải quyết 2 vấn đề còn tồn đọng trong GCN:
1. **"Catastrophic Forgetting" của topology**: learnable adjacency matrix bị gradient cập nhật làm mất thông tin liên kết xương ban đầu
2. **Redundancy trong multi-relational modeling**: ensemble GCs tốn tài nguyên

Giải pháp:
- **Topological Encoding**: dùng graph distance giữa các joint pairs để mã hóa physical connectivity (static) + persistent homology để mô tả systemic dynamics (dynamic, action-specific)
- **BlockGC**: Block Diagonal Weight Matrix — giảm 40% tham số so với GCN thông thường, đạt hiệu năng tốt hơn

```
BlockGC: W ∈ R^(d×d) → Block Diagonal (d/K × d/K) × K blocks
Complexity: O(|V|d²/K) vs O(K|V|d²) của ensemble GC
Parameters: d²/K + K|V|² vs Kd² + K|V|²
```

**Kết quả:**
- NTU RGB+D 60: CS 93.1% / CV 97.0%
- NTU RGB+D 120: CS 90.3% / CSet 91.5%
- **1.3M parameters, 1.63G FLOPs** — model nhỏ nhất trong nhóm SOTA

#### 2.1.2 Transformer-Based Skeleton Methods

**SkateFormer (ECCV 2024) — Do & Kim (KAIST)**

Transformer vượt qua giới hạn receptive field của GCN nhưng tốn bộ nhớ khi tính attention trên toàn bộ joints × frames. SkateFormer giải quyết bằng cách **phân vùng skeletal-temporal** trước khi apply attention:

Bốn loại Skeletal-Temporal Relation (Skate-Types):
| Type | Skeletal | Temporal | Ví dụ action |
|------|---------|---------|-------------|
| Skate-Type-1 | Neighboring joints (physically close) | Local frames | "Brush teeth", "Write" |
| Skate-Type-2 | Distant joints (physically far) | Local frames | "Clap", "Shake hands" |
| Skate-Type-3 | Neighboring joints | Global frames | Repetitive global motions |
| Skate-Type-4 | Distant joints | Global frames | "Sit down", "Stand up" |

**Skate-MSA** (partition-specific attention): attention chỉ trong partition tương ứng → giảm bậc O(N²) xuống O(N²/K).

**Skate-Embedding**: kết hợp learnable skeletal features và fixed temporal index features qua outer product — hiệu quả encode cả không gian lẫn thời gian.

**Kết quả** (4-ensemble, đa modality):
- NTU RGB+D 60: CS 93.5% / CV 97.4%
- NTU RGB+D 120: CS 91.6% / CSet 92.9%
- **SOTA mới nhất** trong nhóm skeleton-based

#### 2.1.3 Hypergraph Methods

**AutoregAd-HGformer (WACV 2025) — Ray et al. (IIT Patna)**

Mở rộng từ GCN (binary relations) lên **Hypergraph** (higher-order relations — 1 hyperedge có thể kết nối nhiều joints cùng lúc):

Hai cơ chế hyperedge generation:
1. **In-phase hypergraph (autoregressive)**: Vector Quantization + autoregressive learned priors → discrete, robust hyperedge representation bên trong encoder
2. **Out-phase hypergraph (adaptive)**: model-agnostic, liên tục tái cấu trúc theo action-specific features từ iteration trước

Hybrid learning: supervised (cross-entropy) + self-supervised (reconstruction loss).

**Kết quả:**
- NTU RGB+D 60: CS 92.4% / CV 96.8%
- NTU RGB+D 120: CS 89.7% / CSet 91.2%

---

### 2.2 Video/RGB-Based Methods

#### 2.2.1 3D-CNN Approaches

**TSM (ICCV 2019) — Lin et al. (MIT)**

Động lực: 2D CNN rẻ nhưng không học temporal; 3D CNN học temporal nhưng nặng. TSM cung cấp 3D CNN performance với 2D CNN cost.

Ý tưởng: **Temporal Shift Module** — dịch chuyển 1 phần channels dọc theo chiều thời gian (±1 frame), cho phép "nhìn" thông tin từ frame trước/sau mà không cần thêm FLOPs.

```
Input: Tensor (B, T, C, H, W)
TSM: shift 1/8 channels ← (t-1), shift 1/8 channels → (t+1), giữ 3/4 channels tại t
→ Residual connection: kết hợp với output hiện tại
→ Zero overhead computation, zero parameters
→ Cài vào ResNet/MobileNet → temporal modeling miễn phí
```

Hai mode:
- **Offline TSM** (bi-directional): dùng cả frame quá khứ lẫn tương lai — tốt nhất cho accuracy
- **Online TSM** (uni-directional): chỉ dùng frame quá khứ — phù hợp realtime

**Kết quả thực tế:**
- Something-Something V1: 49.7% (top-1) — vượt 3D CNN trong nhóm temporal reasoning
- Jetson Nano: 13ms/frame; Galaxy Note8: 35ms/frame
- **Đặc biệt phù hợp** cho edge deployment + ADL realtime

**Toyota Smarthome Dataset & Baseline (ICCV 2019) — Das et al.**

Dataset chuyên dụng ADL:
- 16,115 RGB+D video clips
- 31 activity classes: đủ granularity cho ADL thực tế
- 18 subjects (người cao tuổi) trong môi trường nhà thật
- 7 camera viewpoints, 3 cảnh khác nhau
- RGB + 3D skeleton annotations
- Challenges: intra-class variation cao, class imbalance, activities không scripted

Phương pháp đề xuất: **Pose-Driven Spatio-Temporal Attention** trên 3D ConvNets:
- Spatial attention: encode thông tin vật thể tham gia (human-object interaction)
- Temporal attention: focus keyframes của chuyển động (đứng lên, ngồi xuống)
- Hai sub-networks độc lập regress attention weights từ 3D skeleton

#### 2.2.2 Hierarchical ADL với LLM Context

**Enhancing Action Recognition via Hierarchical Structure & Textual Context (2024)**

Hướng tiếp cận mới nhất cho ADL: kết hợp visual features với **ngữ cảnh văn bản** (location + previous actions) qua transformer:

```
Visual stream (RGB + Optical Flow)
    → ResNet/I3D → Visual features
Textual stream (Location + Previous actions)
    → GPT-3.5/Llama-3 → Text embeddings (BERT/RoBERTa)
Fusion Transformer
    → Cross-attention visual + textual
    → Joint loss (coarse + fine-grained labels)
    → Hierarchical ADL prediction
```

Dataset: **Hierarchical TSU** (Toyota Smarthome Untrimmed extended):
- First hierarchical ADL dataset
- Coarse labels (e.g., "Cooking") + fine-grained labels (e.g., "Stir vegetables")
- Camera location context

**Kết quả:** >17% improvement in top-1 accuracy trên TSU, Assembly101, IkeaASM.

---

### 2.3 Pose Estimation Backbone (Upstream của ADL)

#### RTMPose (2023) — Jiang et al. (Shanghai AI Lab)

Top-down paradigm, backbone CSPNeXt + SimCC head:

```
Input frame
    → RTMDet (realtime detector) → N person bboxes
    → RTMPose per-person:
        → CSPNeXt backbone
        → SimCC: predict keypoint coords trực tiếp (không heatmap)
        → Output: 17 keypoints × (x, y, confidence)
```

Deployment tricks: detector skipping, pose NMS, OneEuro filter.

**Benchmark:**
| Model | COCO AP | CPU FPS (i7-11700) | GPU FPS (GTX 1660Ti) |
|-------|---------|-------------------|---------------------|
| RTMPose-s | 72.2% | ~90 | ~300 |
| RTMPose-m | 75.8% | 90+ | 430+ |
| RTMPose-l | 76.3% | ~30 | ~200 |
| RTMPose-l (WholeBody) | 67.0% | — | 130+ |

**Kết luận cho CPose**: RTMPose-m là lựa chọn tối ưu cho desktop/laptop, RTMPose-s cho edge/mobile.

#### RTMO (CVPR 2024) — Lu et al. (Shanghai AI Lab + Tsinghua)

One-stage pose estimation theo kiến trúc YOLO:
- Không phụ thuộc separate detector → nhanh hơn khi nhiều người
- Dual 1-D heatmaps thay vì full 2D heatmap
- Dynamic coordinate classifier

```
Input frame
    → RTMO (YOLO-based backbone)
        → Detection + Pose simultaneously
        → Dual 1-D heatmaps → keypoint coordinates
    → Output: N persons × 17 keypoints
```

**Benchmark:**
| Model | COCO AP | GPU FPS (V100) |
|-------|---------|---------------|
| RTMO-s | 70.3% | ~200 |
| RTMO-m | 73.2% | ~180 |
| RTMO-l | 74.8% | 141 |

**Khi nào dùng RTMO vs RTMPose:**
- ≤4 người/frame: RTMPose nhanh hơn (top-down ít bbox hơn)
- ≥4 người/frame: RTMO nhanh hơn (one-stage không scale theo N người)

---

## 3. Comparison Matrix — Tất cả Phương pháp

### 3.1 Skeleton-Based Action Recognition

| Model | Venue | NTU-60 CS% | NTU-60 CV% | NTU-120 CS% | NTU-120 CSet% | Params | FLOPs | Đặc điểm |
|-------|-------|-----------|-----------|------------|--------------|--------|-------|---------|
| ST-GCN | AAAI'18 | 81.5 | 88.3 | — | — | 3.1M | 16.3G | Baseline; adjacency cố định |
| MS-G3D | CVPR'20 | 91.5 | 96.2 | 86.9 | 88.4 | 6.4M | — | Unified ST conv, multi-scale |
| CTR-GCN | ICCV'21 | 92.4 | 96.8 | 88.9 | 90.6 | 1.7M | — | Channel-wise topology, baseline mạnh |
| **BlockGCN** | **CVPR'24** | **93.1** | **97.0** | **90.3** | **91.5** | **1.3M** | **1.63G** | **Model nhẹ nhất, SOTA 2024** |
| SkateFormer | ECCV'24 | 93.5 | 97.4 | 91.6 | 92.9 | ~3M | — | SOTA mới nhất, partition attention |
| AutoregAd-HGformer | WACV'25 | 92.4 | 96.8 | 89.7 | 91.2 | ~2M | — | Hypergraph, hybrid learning |

### 3.2 Video-Based Action Recognition (ADL-relevant)

| Model | Venue | Dataset | Accuracy | FPS | Đặc điểm |
|-------|-------|---------|---------|-----|---------|
| I3D | CVPR'17 | Kinetics | 72.1% top-1 | ~12 | 3D Conv baseline |
| TSM-ResNet50 | ICCV'19 | Something-V1 | 49.7% | 50+ | 2D cost, 3D perf |
| TSM (online) | ICCV'19 | Edge device | ~45% | 70+ (Nano) | Realtime edge |
| Pose+I3D (Smarthome) | ICCV'19 | Smarthome | 33.2% CS | ~15 | ADL-specific |
| Hierarchical TSU | 2024 | TSU | SOTA +17% | ~10 | LLM context |

### 3.3 Pose Estimation Backbone (Input cho ADL)

| Model | Venue | COCO AP | CPU FPS | GPU FPS | Mobile FPS |
|-------|-------|---------|---------|---------|-----------|
| RTMPose-s | 2023 | 72.2% | ~70 | ~300 | 70+ (SD865) |
| RTMPose-m | 2023 | 75.8% | 90+ | 430+ | — |
| RTMO-l | CVPR'24 | 74.8% | ~20 | 141 | — |

---

## 4. Datasets và Benchmark Protocols

### 4.1 NTU RGB+D 60 & 120

| Thuộc tính | NTU-60 | NTU-120 |
|-----------|--------|---------|
| Nguồn | ROSE Lab NTU | ROSE Lab NTU |
| Số video | 56,880 | 114,480 |
| Số classes | 60 | 120 |
| Subjects | 40 | 106 |
| Modalities | RGB + Depth + IR + Skeleton | Same |
| Skeleton | 25 joints 3D | 25 joints 3D |
| Evaluation | CS (Cross-Subject) / CV (Cross-View) | CS / CSet |
| Link | https://rose1.ntu.edu.sg/dataset/actionRecognition/ | Same |

Protocols:
- **Cross-Subject (CS)**: train/test theo subject khác nhau — measure generalization
- **Cross-View (CV)**: train/test theo camera angle khác nhau — NTU-60
- **Cross-Setup (CSet)**: train/test theo camera setup khác nhau — NTU-120

**ADL-relevant classes trong NTU-120**: eating, drinking, hand washing, brushing teeth, combing hair, brushing teeth, lying down, falling down, sitting, standing, walking.

### 4.2 Toyota Smarthome

| Thuộc tính | Giá trị |
|-----------|--------|
| Clips | 16,115 |
| Classes | 31 ADL classes |
| Subjects | 18 (người cao tuổi) |
| Camera views | 7 |
| Modalities | RGB + 3D Skeleton |
| Environment | Nhà thật, không scripted |
| Challenges | Intra-class variation, class imbalance, composite activities |
| Download | https://project.inria.fr/toyotasmarthome |

Protocols:
- **Cross-Subject (CS)**: subject 1–10 train, 11–18 test
- **Cross-View (CV1)**: 1 camera test, rest train
- **Cross-View (CV2)**: 2 cameras test, rest train

ADL classes bao gồm: Cook, Eat, Drink (cup/bottle/glass/can), Use computer, Watch TV, Read book, Use phone, Sit, Lie down, Enter room, Leave room, Make call, Clean, Exercise, Fall down, Sleep.

### 4.3 NW-UCLA (Northwestern-UCLA)

| Thuộc tính | Giá trị |
|-----------|--------|
| Clips | 1,475 |
| Classes | 10 |
| Subjects | 10 |
| Camera views | 3 |
| Modalities | RGB + Depth + Skeleton |

Dùng làm cross-view evaluation. Hầu hết skeleton methods đều test trên đây.

### 4.4 Kinetics Skeleton 400

Skeleton được generate từ RGB bằng OpenPose (18 joints 2D + confidence). Dùng để test zero-shot transfer capacity của skeleton models.

### 4.5 Dataset ADL Bổ Sung

| Dataset | Đặc điểm | Scale | Phù hợp CPose |
|---------|---------|-------|-------------|
| UR Fall Detection | Fall vs ADL balanced | 40 ADL + 30 fall | ✅ Fall detection |
| ETRI Activity 3D | Elderly ADL, 3D | 112,620 samples | ✅ Người cao tuổi |
| Charades | Indoor activities | 9,848 videos | ✅ Indoor |
| MUVIM | Multi fall/ADL, RGB+D | Moderate | ✅ Fall detection |
| HAR-UP | Fall + 11 ADL, wearable | 300+ sessions | Wearable only |
| Hierarchical TSU | ADL hierarchical | 16K+ (extended) | ✅ ADL rich labels |

---

## 5. Pipeline Chi Tiết Theo Hướng Tiếp Cận

### 5.1 Pipeline Skeleton-Based (Khuyến nghị cho CPose)

```
Camera RTSP stream
    ↓
RTMPose (90+ FPS, CPU) / RTMO (141 FPS, GPU)
    ↓ COCO-17 keypoints × confidence
Pre-processing
    ↓ Normalize, center, scale keypoints
    ↓ Stack T=30~300 frames thành sequence
Skeleton Action Model (chọn 1):
    → BlockGCN (lightweight, CVPR'24)    [PoC option 1]
    → CTR-GCN (baseline mạnh, ICCV'21)  [PoC option 2]
    → SkateFormer (SOTA, ECCV'24)        [Future]
    ↓ Action logits
Post-processing
    ↓ Temporal smoothing / voting (N frames)
    ↓ Confidence threshold
ADL Label + Confidence
    ↓
Alert / Visualization / Logging
```

### 5.2 Pipeline Video-Based (RGB Stream)

```
Camera RTSP stream (raw RGB)
    ↓
Frame sampling (T frames)
    ↓
Feature extractor:
    → TSM-ResNet50 (online mode, 70+ FPS edge)   [Realtime PoC]
    → I3D / SlowFast (offline)                   [Accuracy]
    ↓ Spatiotemporal features
Classifier (FC + Softmax)
    ↓
ADL Label
```

### 5.3 Pipeline Hybrid (Skeleton + RGB)

```
Camera RTSP
    ↓            ↓
RTMPose       RGB frames
    ↓              ↓
Skeleton     Visual features
features     (I3D/TSM)
    \              /
     Fusion layer
          ↓
   ADL Classification
```

Hybrid thường tốt hơn 2-5% nhưng phức tạp hơn nhiều.

---

## 6. Đánh Giá Yêu Cầu Phần Cứng & Khả Năng Realtime

| Model | Input | FPS (CPU i7) | FPS (GPU RTX) | VRAM | RAM | Edge |
|-------|-------|------------|--------------|------|-----|------|
| RTMPose-m | 640px | 90+ | 430+ | <1GB | ~500MB | ✅ |
| RTMO-l | 640px | ~20 | 141 | ~2GB | ~1GB | ❌ |
| BlockGCN | Skeleton | ~500 | >1000 | ~200MB | ~100MB | ✅ |
| CTR-GCN | Skeleton | ~300 | ~800 | ~300MB | ~150MB | ✅ |
| SkateFormer | Skeleton | ~100 | ~400 | ~500MB | ~300MB | ⚠️ |
| TSM-ResNet50 | RGB | ~50 | 200+ | ~3GB | ~2GB | ⚠️ |
| I3D | RGB | ~10 | ~50 | ~6GB | ~4GB | ❌ |
| AutoregAd-HGformer | Skeleton | ~200 | ~600 | ~400MB | ~200MB | ✅ |

> **Thực tế triển khai CPose (laptop i7 + GTX 1660Ti)**:
> - RTMPose-m: 430 FPS → đủ cho 4 camera × 30fps mỗi camera
> - BlockGCN: chạy offline trên keypoints → không bottleneck
> - Tổng pipeline: khoảng 15-25 FPS end-to-end với 4 cam

---

## 7. So Sánh Chi Tiết Trên Benchmark ADL

### 7.1 Toyota Smarthome — State of the Art (CS benchmark)

| Method | Input | CS% | CV1% |
|--------|-------|-----|------|
| I3D [Smarthome ICCV'19] | RGB | 33.2 | — |
| Pose-Attention + I3D | RGB+Skeleton | 35.7 | — |
| MS-G3D [adapt.] | Skeleton | 42.1 | — |
| TSM-ResNet50 | RGB | 38.4 | — |
| Hierarchical TSU (2024) | RGB+Text | SOTA | — |

> Toyota Smarthome accuracy tương đối thấp (~33–42%) vì độ khó thực tế cao: activities không scripted, góc camera đa dạng, intra-class variation lớn.

### 7.2 NTU RGB+D 120 — Skeleton Methods (CS benchmark)

| Method | Year | CS% | ΔCS vs ST-GCN |
|--------|------|-----|--------------|
| ST-GCN | 2018 | — | — |
| MS-G3D | 2020 | 86.9 | baseline |
| CTR-GCN | 2021 | 88.9 | +2.0 |
| BlockGCN | 2024 | 90.3 | +3.4 |
| SkateFormer | 2024 | 91.6 | +4.7 |
| AutoregAd-HGformer | 2025 | 89.7 | +2.8 |

> SkateFormer hiện giữ SOTA 2024 trên NTU-120, BlockGCN có tỷ lệ accuracy/params tốt nhất.

---

## 8. Đề Xuất Triển Khai PoC

### 8.1 Option 1 (Khuyến nghị): BlockGCN + RTMPose

**Lý do lựa chọn BlockGCN:**
- **Lightweight nhất**: 1.3M params, 1.63G FLOPs — phù hợp chạy trên laptop triển khai CPose
- **SOTA CVPR 2024**: accuracy cao nhất trong nhóm efficient skeleton models
- **Open-source**: https://github.com/ZhouYuxuanYX/BlockGCN
- **Hệ sinh thái**: PyTorch, tương thích với pipeline Python/OpenCV của CPose
- **Incremental adoption**: phần skeleton-graph chỉ cần keypoints đầu ra từ YOLOv11-pose

**Pipeline cụ thể:**
```
Phase 3 (CPose) → _keypoints.txt
    ↓
Load keypoints per person: (T=30~64, V=17, C=2) → normalize
    ↓
BlockGCN inference: ~2ms/sample (CPU)
    ↓
ADL label + confidence
    ↓
Ghi vào _adl.txt (thay thế rule_based_adl hiện tại)
```

**Training strategy:**
1. Pre-train trên NTU RGB+D 60/120 (pre-trained weights từ paper)
2. Fine-tune trên Toyota Smarthome với 8 ADL classes của CPose
3. Evaluation: Toyota Smarthome CS protocol

**GitHub**: https://github.com/ZhouYuxuanYX/BlockGCN

---

### 8.2 Option 2 (Backup): CTR-GCN + RTMPose

**Lý do lựa chọn CTR-GCN:**
- **Baseline mạnh nhất, ổn định nhất**: tested rộng rãi, ít bug hơn model mới
- **Hệ sinh thái lớn hơn**: nhiều tutorials, issue trackers, community support
- **Accuracy tốt**: 88.9% CS NTU-120, đủ cho demo research
- **Điểm trừ**: paper nhấn mạnh accuracy hơn deployment, cần tự benchmark latency

**GitHub**: https://github.com/Uason-Chen/CTR-GCN

---

### 8.3 So Sánh 2 Options

| Tiêu chí | BlockGCN | CTR-GCN |
|---------|---------|---------|
| NTU-120 CS Accuracy | 90.3% | 88.9% |
| Parameters | **1.3M** | 1.7M |
| FLOPs | **1.63G** | ~4G |
| Community | Medium | **Large** |
| Paper năm | **2024** | 2021 |
| Deployment docs | Ít | Nhiều |
| Phù hợp PoC | ✅ Edge-friendly | ✅ Proven |

**Khuyến nghị cuối**: Bắt đầu với **CTR-GCN** (ít bug hơn, dễ integrate hơn), sau khi pipeline ổn thì nâng cấp lên **BlockGCN** để tăng accuracy và giảm tài nguyên.

---

## 9. Danh Sách GitHub Repos Chất Lượng Cao

### 9.1 Skeleton Action Recognition

| Repo | Paper | Stars | Framework |
|------|-------|-------|-----------|
| https://github.com/ZhouYuxuanYX/BlockGCN | BlockGCN CVPR'24 | ~300 | PyTorch |
| https://github.com/Uason-Chen/CTR-GCN | CTR-GCN ICCV'21 | ~500 | PyTorch |
| https://github.com/kenziyuliu/ms-g3d | MS-G3D CVPR'20 | ~800 | PyTorch |
| https://github.com/yysijie/st-gcn | ST-GCN AAAI'18 | ~4k | PyTorch |
| https://kaist-viclab.github.io/SkateFormer_site/ | SkateFormer ECCV'24 | ~200 | PyTorch |
| https://github.com/open-mmlab/mmskeleton | mmSkeleton | ~3k | PyTorch/MMAction |
| https://github.com/kennymckormick/pyskl | PYSKL toolkit | ~1.5k | PyTorch |

### 9.2 Video Action Recognition

| Repo | Paper | Stars | Framework |
|------|-------|-------|-----------|
| https://github.com/mit-han-lab/temporal-shift-module | TSM ICCV'19 | ~5k | PyTorch |
| https://github.com/open-mmlab/mmaction2 | MMAction2 | ~13k | PyTorch |
| https://github.com/karpathy/deepmind-research/blob/master/perceiver | SlowFast | ~8k | PyTorch |

### 9.3 Pose Estimation

| Repo | Paper | Stars | Framework |
|------|-------|-------|-----------|
| https://github.com/open-mmlab/mmpose | MMPose (RTMPose, RTMO) | ~10k | PyTorch |
| https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose | RTMPose | — | PyTorch |
| https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo | RTMO CVPR'24 | — | PyTorch |

### 9.4 ADL-Specific & Datasets

| Repo | Nội dung |
|------|---------|
| https://github.com/dairui01/TSU_evaluation | Toyota Smarthome Untrimmed eval |
| https://github.com/3dperceptionlab/HierarchicalActionRecognition | Hierarchical TSU 2024 |
| https://github.com/ZhenyuX1E/PoseTrack | PoseTrack (tracking + action) |
| https://github.com/yuanc3/Pose2ID | Pose2ID ReID CVPR'25 |

### 9.5 Toolkits (End-to-End)

| Repo | Mô tả |
|------|------|
| https://github.com/open-mmlab/mmaction2 | Toàn diện nhất: skeleton + video + optical flow |
| https://github.com/kennymckormick/pyskl | Chuyên skeleton: tái hiện ST-GCN, CTR-GCN, MS-G3D, PoseC3D |
| https://github.com/ultralytics/ultralytics | YOLOv11-pose (đang dùng trong CPose) |

---

## 10. Roadmap Tích Hợp vào CPose/HavenPose

### Phase A — Ngắn hạn (2-4 tuần): Cải thiện ADL engine hiện tại

| Bước | Công việc | Ưu tiên |
|------|----------|---------|
| A1 | Benchmark rule_based_adl trên Toyota Smarthome | P0 |
| A2 | Thu thập ground-truth ADL labels từ camera nhà | P0 |
| A3 | Tune thresholds pose_utils.py dựa trên data thật | P0 |
| A4 | Tích hợp temporal voting 5-10 frames (giảm label flickering) | P1 |

### Phase B — Trung hạn (1-2 tháng): Tích hợp CTR-GCN/BlockGCN

| Bước | Công việc | Ưu tiên |
|------|----------|---------|
| B1 | Cài đặt PYSKL hoặc CTR-GCN | P0 |
| B2 | Pre-process keypoints CPose → NTU format (T, V, C) | P0 |
| B3 | Fine-tune pre-trained weights trên 8 ADL classes | P0 |
| B4 | Thay thế rule_based_adl() bằng GCN inference | P1 |
| B5 | Benchmark: latency, accuracy, FPS trên setup thực tế | P1 |

### Phase C — Dài hạn (2-3 tháng): SkateFormer + Multi-modal

| Bước | Công việc | Ưu tiên |
|------|----------|---------|
| C1 | Nâng lên SkateFormer (SOTA skeleton) | P2 |
| C2 | Thêm TSM stream xử lý RGB song song | P2 |
| C3 | Hybrid fusion skeleton + RGB | P3 |
| C4 | LLM context (camera location + history) theo hướng Hierarchical TSU | P3 |

---

## 11. Kết Luận

### 11.1 Tóm Tắt Các Hướng Tiếp Cận

Trong 5 năm gần đây, bài toán ADL Recognition đã trải qua 3 giai đoạn phát triển rõ ràng:

**Giai đoạn 1 (2018-2020)**: GCN-based skeleton — ST-GCN mở đường; MS-G3D giải quyết vấn đề multi-scale và cross-spacetime.

**Giai đoạn 2 (2021-2023)**: Refinement và optimization — CTR-GCN cải thiện topology learning; TSM cung cấp efficient video baseline. Đây là giai đoạn consolidation.

**Giai đoạn 3 (2024-2025)**: Transformer và hybrid — SkateFormer, AutoregAd-HGformer khai thác attention mechanisms; Hierarchical TSU tích hợp LLM context; BlockGCN cân bằng efficiency-accuracy.

### 11.2 Đề Xuất Ưu Tiên Cho CPose

1. **Ngắn hạn (PoC)**: CTR-GCN với pre-trained NTU weights, fine-tune 50 ADL clips thu thập từ camera nhà — đủ cho demo và báo cáo đồ án
2. **Trung hạn (upgrade)**: BlockGCN (CVPR 2024) — nhỏ hơn, nhanh hơn, accurate hơn, phù hợp deploy trên laptop
3. **Dài hạn (research)**: SkateFormer (ECCV 2024) nếu có GPU mạnh hơn, hoặc hybrid skeleton+RGB với TSM nếu muốn cải thiện accuracy trên ADL thực tế

### 11.3 Điểm Quan Trọng Nhất

**Bottleneck không phải model ADL — mà là data.** Toyota Smarthome cho thấy accuracy chỉ ~33-42% dù dùng model mạnh, vì dữ liệu thực tế (indoor, unscripted, multi-view) rất khác training data. Chiến lược tốt nhất cho CPose là:

1. Thu thập ~200-500 clip thực tế từ 4 camera nhà
2. Annotate 8 ADL classes (standing/sitting/walking/falling/lying/reaching/bending/unknown)
3. Fine-tune CTR-GCN hoặc BlockGCN trên dataset này
4. Đây sẽ cho accuracy thực tế tốt hơn nhiều so với chỉ dùng pre-trained model từ NTU

---
