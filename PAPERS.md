# PAPERS.md — Literature Review for CPose

> **Project:** CPose — TFCS-PAR Multi-Camera Human Activity Pipeline  
> **Maintainer:** Nguyễn Bá Thành  
> **Scope:** All papers used for research, implementation, and citation in CPose paper.  
> **Coverage:** 2018–2026, organized by module relevance.

---

## Table of Contents

1. [Pose Estimation](#1-pose-estimation)
2. [Skeleton-Based Action Recognition (ADL)](#2-skeleton-based-action-recognition-adl)
3. [Video-Based Action Recognition](#3-video-based-action-recognition)
4. [Multi-Person Tracking (MOT)](#4-multi-person-tracking-mot)
5. [Person Re-Identification (ReID)](#5-person-re-identification-reid)
6. [Datasets](#6-datasets)
7. [Citation Priority for CPose Paper](#7-citation-priority)

---

## 1. Pose Estimation

### 1.1 2D Real-Time Pose Estimation

#### RTMPose (2023) ⭐ Production-ready — Used in CPose
- **Paper:** RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose
- **arXiv:** https://arxiv.org/abs/2303.07399
- **GitHub:** https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- **Dataset:** COCO Keypoints, COCO-WholeBody, Body8
- **Key results:**
  - RTMPose-m: **75.8% AP COCO** | **90+ FPS CPU i7** | **430+ FPS GTX 1660Ti**
  - RTMPose-s: 72.2% AP | 70+ FPS Snapdragon 865
- **CPose use:** Primary pose backbone. RTMPose-m for desktop, RTMPose-s for edge.
- **Architecture:** CSPNeXt backbone + SimCC head (no heatmap, direct coordinate prediction)

#### RTMO (CVPR 2024) ⭐ One-stage alternative
- **Paper:** RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation
- **GitHub:** https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo
- **Dataset:** COCO val2017, CrowdPose
- **Key results:**
  - RTMO-l: **74.8 AP COCO val2017** | **141 FPS V100**
  - CrowdPose: 73.2 AP
- **CPose use:** Preferred when ≥4 persons/frame (one-stage doesn't scale with N persons)
- **Architecture:** YOLO-based, dual 1-D heatmaps, dynamic coordinate classifier

#### ViTPose++ (TPAMI 2023)
- **arXiv:** https://arxiv.org/abs/2212.04246
- **GitHub:** https://github.com/ViTAE-Transformer/ViTPose
- **Key results:** 81.1 AP MS COCO test-dev | 20M–1B parameters
- **Note:** Foundation model for pose, overkill for CPose realtime but useful for research benchmark

#### DWPose (ICCV 2023 Workshop)
- **GitHub:** https://github.com/IDEA-Research/DWPose
- **Dataset:** COCO-WholeBody
- **Note:** Two-stage distillation for whole-body pose; useful if face/hand keypoints needed

---

### 1.2 3D Pose Estimation

#### MotionBERT (ICCV 2023) ⭐ Used as ReID/ADL literature anchor
- **arXiv:** https://arxiv.org/abs/2210.06551
- **GitHub:** https://github.com/Walter0807/MotionBERT
- **Key results:**
  - 3D HPE: MPJPE reduction on Human3.6M
  - Action recognition: **97.2% Top-1 NTU-60 xsub** (cite in paper)
  - Mesh recovery: SOTA on 3DPW
- **Architecture:** Unified motion pretraining; 1–2 linear layers for downstream tasks

#### PoseFormerV2 (CVPR 2023)
- **arXiv:** https://arxiv.org/abs/2303.17422
- **GitHub:** https://github.com/QitaoZhao/PoseFormerV2
- **Key results:** MPJPE 46.0mm Human3.6M
- **Note:** Frequency-domain lifting for robustness to noisy 2D joints

#### UniPose (CVPR 2025 Highlight)
- **GitHub:** https://github.com/VIPL-VISMOD/UniPose
- **Note:** Multi-modal LLM for 7 pose tasks; future direction for CPose

#### PersPose (ICCV 2025)
- **GitHub:** https://github.com/KenAdamsJoseph/PersPose
- **Note:** Perspective encoding for camera-aware 3D HPE

---

### 1.3 Pose Estimation Surveys

| Survey | Venue | Link | Scope |
|---|---|---|---|
| Deep Learning-Based HPE: A Survey (Zheng et al.) | ACM CSUR 2023 | https://dl.acm.org/doi/10.1145/3603618 | 260+ papers, 2D/3D |
| Efficient Monocular HPE Survey | IEEE Access 2024 | https://doi.org/10.1109/access.2024.3399222 | Edge/mobile focus |
| Survey on DL for 2D and 3D HPE | AI Review 2026 | https://www.scribd.com/document/984031556/ | Latest trends, LLM/diffusion |

---

## 2. Skeleton-Based Action Recognition (ADL)

### 2.1 GCN-Based (CPose Research Target)

#### ST-GCN (AAAI 2018) — Baseline
- **Paper:** Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
- **Key results:** NTU-60 CS: 81.5%, CV: 88.3%
- **Architecture:** Fixed adjacency matrix; centripetal/centrifugal/root partitioning
- **Input:** (C=3, T=300, V=18, M=2) tensor
- **Limitation:** Adjacency is static; cannot model long-range joint relations

#### MS-G3D (CVPR 2020)
- **Key results:** NTU-60 CS: 91.5%, CV: 96.2% | NTU-120 CS: 86.9%
- **Innovation:** Disentangled multi-scale aggregation + cross-spacetime G3D module

#### CTR-GCN (ICCV 2021) — Strong Baseline
- **GitHub:** https://github.com/Uason-Chen/CTR-GCN (community)
- **Key results:** NTU-60 CS: 92.4%, CV: 96.8% | NTU-120 CS: 88.9% | NW-UCLA: 96.5%
- **Innovation:** Channel-wise Topology Refinement (per-channel adjacency learning)
- **CPose use:** Reference baseline for ADL skeleton approach

#### BlockGCN (CVPR 2024) ⭐ Recommended for CPose ADL upgrade
- **GitHub:** https://github.com/firework8/BlockGCN (community)
- **Key results:** NTU-60 CS: 93.1%, CV: 97.0% | NTU-120 CS: 90.3%, CSet: 91.5%
- **Efficiency:** **1.3M parameters, 1.63G FLOPs** — smallest SOTA model
- **Innovation:** Block diagonal weight matrix; topological encoding with persistent homology
- **CPose use:** Recommended as first learned ADL model (lightweight, SOTA 2024)

#### HD-GCN (ICCV 2023)
- **GitHub:** https://github.com/Jho-Yonsei/HD-GCN
- **Key results:** SOTA on NTU-60/120/NW-UCLA (6-way ensemble)
- **Innovation:** Hierarchically decomposed graph

#### Hyper-GCN (ICCV 2025)
- **Link:** https://iccv.thecvf.com/virtual/2025/poster/2047
- **Key results:** NTU-120 X-Sub: 90.5%, X-Set: 91.7%
- **Innovation:** Adaptive hypergraph convolution

---

### 2.2 Transformer-Based Skeleton Methods

#### SkateFormer (ECCV 2024) ⭐ Current SOTA
- **arXiv:** https://arxiv.org/abs/2403.09508
- **GitHub:** https://github.com/KAIST-VICLab/SkateFormer
- **Key results:** NTU-60 CS: 93.5%, CV: 97.4% | NTU-120 CS: 91.6%, CSet: 92.9%
- **Innovation:** Skeletal-temporal partitioning (4 Skate-Types) → reduces O(N²) attention
- **CPose use:** SOTA target for publication; compare against BlockGCN baseline

#### AutoregAd-HGformer (WACV 2025)
- **Affiliation:** IIT Patna
- **Key results:** NTU-60 CS: 92.4% | NTU-120 CS: 89.7%
- **Innovation:** Autoregressive + adaptive hyperedge generation; hybrid supervised/self-supervised

#### ProtoGCN (CVPR 2025 Highlight)
- **GitHub:** https://github.com/firework8/ProtoGCN
- **Innovation:** Prototype-based perspective for skeleton recognition

---

### 2.3 ADL-Specific Methods

#### π-ViT / PI-ViT (CVPR 2024) ⭐ Toyota Smarthome SOTA — Cite in CPose
- **arXiv:** https://arxiv.org/abs/2311.18840
- **GitHub:** https://github.com/dominickrei/pi-vit
- **Key results:**
  - Toyota Smarthome CS: **72.9% mCA** (SOTA)
  - NTU-60 Top-1: **94.0%**
- **Architecture:** Pose-Induced Video Transformer; 2D-SIM + 3D-SIM plug-in modules
- **CPose use:** Key literature anchor for ADL on surveillance data

#### SKI Models (AAAI 2025)
- **GitHub:** https://github.com/thearkaprava/SKI-Models
- **Innovation:** Skeleton-induced vision-language embeddings for ADL

#### Pose-guided Token Selection (2025)
- **Link:** https://www.sciencedirect.com/science/article/pii/S0262885625002744
- **Innovation:** Token selection from pose for ADL efficiency

#### ADL Recognition via Multi-Modal Deep Learning (arXiv 2026)
- **arXiv:** https://arxiv.org/abs/2603.04509
- **Innovation:** I3D + GCN + object detection with pose-driven temporal attention

#### Exploring ADL Among Elderly (IJORAS 2025)
- **Note:** Custom dataset, 5 ADLs, 20 subjects; CNN best at 82.86%
- **CPose relevance:** Elderly monitoring use case alignment

#### Toyota Smarthome Dataset & Baseline (ICCV 2019)
- **Link:** https://project.inria.fr/toyotasmarthome
- **Scale:** 16,115 clips, 31 ADL classes, 18 elderly subjects, 7 cameras
- **Protocol:** CS, CV1, CV2
- **CPose use:** Primary ADL benchmark reference; cite baseline numbers

#### Enhancing Action Recognition (Hierarchical TSU, 2024)
- **Innovation:** Visual + LLM text context fusion; +17% improvement on TSU
- **Note:** Future direction for context-aware ADL

---

## 3. Video-Based Action Recognition

#### TSM (ICCV 2019) — Edge Realtime Baseline
- **Paper:** Temporal Shift Module for Efficient Video Understanding
- **Key results:**
  - Something-Something V1: 49.7% top-1
  - Jetson Nano: 13ms/frame; Galaxy Note8: 35ms/frame
- **Innovation:** Zero-overhead temporal modeling (channel shifting, no extra FLOPs)
- **CPose use:** RGB backup if skeleton not available; online mode for realtime

---

## 4. Multi-Person Tracking (MOT)

### 4.1 Core Trackers

| Tracker | Venue | arXiv | HOTA (MOT17) | CPose Use |
|---|---|---|---|---|
| **ByteTrack** | ECCV 2022 | https://arxiv.org/abs/2110.06864 | 63.1 | **Default tracker** |
| OC-SORT | CVPR 2023 | https://arxiv.org/abs/2203.14360 | ~63.9 | Occlusion robustness |
| Deep OC-SORT | 2023 | https://arxiv.org/abs/2302.11813 | 64.9 | Appearance + motion |
| BoT-SORT | 2022 | https://github.com/NirAharon/BoT-SORT | 65.0 | Camera-motion aware |
| StrongSORT | 2023 | https://github.com/dyhBUPT/StrongSORT | ~63+ | Strong baseline |
| OmniTrack | CVPR 2025 | https://arxiv.org/abs/2503.04565 | — | 360° cameras (future) |

**BoxMOT (2023–2026):** https://github.com/mikel-brostrom/boxmot  
Multi-tracker framework: DeepOCSORT, BoTSORT, StrongSORT, ByteTrack + ReID models (CLIPReID, OSNet, LightMBN)

---

## 5. Person Re-Identification (ReID)

### 5.1 Pose-Aware ReID

#### Keypoint Promptable ReID / KPR (ECCV 2024) ⭐ Occluded ReID SOTA
- **arXiv:** https://arxiv.org/abs/2407.18112
- **GitHub:** https://github.com/VlSomers/keypoint_promptable_reidentification
- **Dataset:** Occluded-PoseTrack-ReID, Market-1501, Occluded-Duke
- **Key results:** SOTA on all occluded ReID benchmarks
- **Innovation:** Semantic keypoints as prompts; solves Multi-Person Ambiguity (MPA)
- **CPose use:** Architecture reference for occluded person handling

#### Pose-Transfer ReID (CVPR classic)
- **Key results:** Market-1501 Rank-1: 87.65%, mAP: 68.92%
- **CPose use:** Baseline for body feature extraction

#### Pose2ID (CVPR 2025)
- **Note:** Pose-based identity representation; relevant for no-face scenarios

---

### 5.2 Cross-Camera / Multi-Camera ReID

#### MCPT (Multi-Camera People Tracking)
- **Key results:** **86.76% IDF1 AI City 2023** (cite in paper)
- **CPose use:** Literature anchor for cross-camera tracking performance

#### Market-1501, DukeMTMC-reID, CUHK03, MARS — Standard ReID Datasets
- Market-1501: 1,501 identities, 6 cameras
- DukeMTMC-reID: 1,404 identities, 8 cameras
- MARS: Video-based ReID, 1,261 identities

---

### 5.3 Appearance Features

| Method | Backbone | Dim | CPose Use |
|---|---|---|---|
| ResNet-50 + IBN | ResNet | 2048 | Body feature baseline |
| OSNet | Lightweight | 512 | Edge-friendly body feature |
| LightMBN | Lightweight | 512 | BoxMOT integration |
| CLIP-ReID | ViT-B/16 | 512 | Zero-shot transfer |

---

## 6. Datasets

### 6.1 Action Recognition Datasets

| Dataset | Classes | Subjects | Modalities | Eval Protocol | Link |
|---|---|---|---|---|---|
| **NTU RGB+D 60** | 60 | 40 | RGB+D+IR+Skel | CS / CV | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| **NTU RGB+D 120** | 120 | 106 | Same | CS / CSet | Same |
| **Toyota Smarthome** | 31 ADL | 18 elderly | RGB+Skel | CS / CV1 / CV2 | https://project.inria.fr/toyotasmarthome |
| NW-UCLA | 10 | 10 | RGB+D+Skel | Cross-view | — |
| ETRI Activity 3D | ADL elderly | — | RGB+D | — | — |
| UR Fall Detection | Fall + ADL | — | RGB+D | — | — |
| Charades | Indoor | — | RGB | — | — |

### 6.2 Pose Estimation Datasets

| Dataset | Keypoints | Scale | Link |
|---|---|---|---|
| **MS COCO Keypoints** | 17 COCO | 200K+ images | https://cocodataset.org |
| Human3.6M | 17 | 3.6M frames | http://vision.imar.ro/human3.6m/ |
| MPII | 16 | 25K images | http://human-pose.mpi-inf.mpg.de/ |
| OCHuman | 17 | 13,360 images (heavy occlusion) | — |

### 6.3 ReID Datasets

| Dataset | Identities | Cameras | Link |
|---|---|---|---|
| Market-1501 | 1,501 | 6 | — |
| DukeMTMC-reID | 1,404 | 8 | — |
| MARS | 1,261 | 6 | — |
| Occluded-PoseTrack-ReID | — | — | KPR paper |

### 6.4 CPose Self-Collected Dataset (TODO)

| Field | Value |
|---|---|
| Cameras | cam01, cam02, cam03, cam04 |
| Topology | cam1→cam2→cam3→elevator→cam4 |
| Persons | TODO |
| Global IDs | TODO |
| Total clips | TODO |
| Total frames | TODO |
| ADL classes | 8 |
| Blind-zone scenarios | elevator, room, door |
| Clothing-change events | TODO |
| FPS | 30 |
| Resolution | TODO |

---

## 7. Citation Priority for CPose Paper

### 7.1 Must-Cite (Core Claims)

| Paper | Reason to Cite |
|---|---|
| RTMPose (2023) | Pose backbone used in system |
| RTMO (CVPR 2024) | Alternative pose backbone |
| ByteTrack (ECCV 2022) | Tracking algorithm |
| π-ViT (CVPR 2024) | ADL SOTA on Smarthome CS (72.9%) |
| MotionBERT (ICCV 2023) | NTU-60 skeleton benchmark (97.2%) |
| MCPT | Cross-camera IDF1 reference (86.76%) |
| Toyota Smarthome (ICCV 2019) | ADL dataset used |

### 7.2 Should-Cite (Related Work Section)

| Paper | Section |
|---|---|
| DeepSORT | §2 Related: tracking |
| BoTSORT, StrongSORT | §2 Related: tracking |
| ST-GCN (AAAI 2018) | §2 Related: skeleton ADL |
| CTR-GCN (ICCV 2021) | §2 Related: skeleton ADL |
| BlockGCN (CVPR 2024) | §2 Related: skeleton ADL |
| SkateFormer (ECCV 2024) | §2 Related: skeleton ADL SOTA |
| KPR (ECCV 2024) | §2 Related: occluded ReID |
| Pose-Transfer ReID | §2 Related: pose-aware ReID |
| TSM (ICCV 2019) | §2 Related: video action recognition |

### 7.3 Benchmark Numbers to Report in Paper (All Must Come from Logs)

```
Your system — run from data/outputs/.../metrics.json
Literature (DO NOT CLAIM AS YOURS):
  RTMPose-m:   75.8 AP COCO | 90+ FPS i7 | 430+ FPS GTX 1660Ti
  RTMO-l:      74.8 AP COCO val2017 | 141 FPS V100
  MotionBERT:  97.2% Top-1 NTU-60 xsub
  π-ViT:       72.9% mCA Smarthome CS | 94.0% Top-1 NTU-60
  MCPT:        86.76% IDF1 AI City 2023
  BlockGCN:    93.1% NTU-60 CS | 1.3M params | 1.63G FLOPs
  SkateFormer: 93.5% NTU-60 CS | 91.6% NTU-120 CS
  ByteTrack:   63.1 HOTA MOT17
```

---

## 8. Research Roadmap

### Phase 1 (Current — PoC)
- Rule-based ADL baseline
- YOLO11n detection + pose
- ByteTrack tracking
- Time-topology Global ID (TFCS-PAR)
- Terminal pipeline + JSON output

### Phase 2 (Metric Collection)
- Annotate test set: bbox + global_id + ADL labels
- Run evaluation module → get ground-truth metrics
- Write benchmark tables

### Phase 3 (Model Upgrade)
- Replace rule-based ADL with BlockGCN
- Compare: rule-based vs. BlockGCN vs. CTR-GCN
- Add pose/gait embedding to ReID score

### Phase 4 (Paper Submission)
- Full ablation (8 variants)
- Transition window ablation
- Clothing-change benchmark
- Runtime table (CPU / GPU / Jetson)
- Write paper with logged numbers only
