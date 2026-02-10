# HAVEN Backend & Frontend Architecture - Restructured

## 🎯 RESTRUCTURE OVERVIEW

### Backend được tổ chức theo 3 layers:

**Layer 1: INPUT LAYER**
- (13) Video Frame Input Handler: Nhận frames từ Camera 4

**Layer 2: STORAGE & PROCESSING LAYER**  
- ⚡ FAISS Vector Database (176-dim vectors, HNSW index)
- 💾 SQLite Relational Database (Person metadata, Events, Trajectories)
- (14) Event Processing Engine (ADL analysis, Fall detection)

**Layer 3: API & OUTPUT LAYER**
- 🔌 REST API (GET /persons, /events, /stats)
- 🌐 WebSocket (/ws/events, /ws/frames, /ws/adl)
- 🎥 Stream API (MJPEG, HLS)
- 📋 CSV Logger
- 🎥 Media Storage
- 📱 Telegram Bot

### Frontend được tổ chức theo features:

**Group 1: MONITORING (Real-time)**
- (15) 📺 Live Multi-Camera View
- (17) 📋 Real-time Event Timeline  
- (20) 🔔 Alert Center

**Group 2: ANALYTICS & INSIGHTS**
- (16) 👤 Person ID Cards
- (18) 🗺️ Trajectory Heatmap
- (19) 📊 Analytics Dashboard

**Group 3: UTILITIES**
- 💾 Export (CSV/PDF/JSON)
- ⚙️ Settings Panel
- 🔍 Search & Filter

---

## 📐 LOGICAL FLOW

```
Camera 4 Output (12)
    ↓
┌─────────────────────────────────────────┐
│  BACKEND LAYER 1: INPUT                 │
│  (13) Video Frame Input Handler         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  BACKEND LAYER 2: STORAGE & PROCESSING  │
│  ├─ ⚡ FAISS Vector DB                   │
│  ├─ 💾 SQLite Database                   │
│  └─ (14) Event Processing Engine        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  BACKEND LAYER 3: API & OUTPUT          │
│  ├─ REST API + WebSocket                │
│  ├─ Stream API                           │
│  ├─ CSV Logger                           │
│  ├─ Media Storage                        │
│  └─ Telegram Bot                         │
└─────────────────────────────────────────┘
    ↓ (WebSocket + REST)
┌─────────────────────────────────────────┐
│  FRONTEND: User Interface               │
│  ┌───────────────────────────────────┐  │
│  │ Monitoring (15, 17, 20)           │  │
│  ├───────────────────────────────────┤  │
│  │ Analytics (16, 18, 19)            │  │
│  ├───────────────────────────────────┤  │
│  │ Utilities (Export, Settings)      │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## 🔧 KEY IMPROVEMENTS

### ✅ Backend Logic
1. **Clear hierarchy**: Input → Storage/Processing → API
2. **Vector DB inside Backend**: Integrated với Event Processor
3. **Unified API layer**: REST + WebSocket + Stream trong một nhóm
4. **Output services grouped**: CSV, Media, Telegram cùng layer với API

### ✅ Frontend Logic  
1. **Feature-based grouping**: Monitoring vs Analytics vs Utilities
2. **Clear data flow**: All connect to Backend API layer
3. **User roles**: 
   - Normal User: Monitoring features (15, 17, 20)
   - Expert User: + Analytics (16, 18, 19) + Settings

### ✅ Clean Connections
- Camera 4 → Backend Input (1 arrow)
- Backend API ↔ Frontend (bidirectional WebSocket)
- Vector DB ↔ Event Processor (internal backend flow)

---

## 📝 IMPLEMENTATION NOTES

Trong file drawio nên:
1. Backend nằm trong 1 box lớn với 3 sections ngang (Layer 1-2-3)
2. Frontend nằm box riêng bên phải, chia 3 groups
3. Vector DB nằm TRONG Backend Layer 2
4. Performance Benchmark riêng ở dưới cùng
