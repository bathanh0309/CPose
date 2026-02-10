# Backend Arrow Logic - Summary of Changes

## ✅ Đã sửa (Completed):

### Arrow 1: Input Handler ↔ Vector DB
- **Query embeddings**: Input Handler query Vector DB để match faces/bodies
- **Store new embeddings**: Bidirectional - DB cũng nhận embeddings mới từ cameras

## 🔧 Cần thêm (Recommended):

### Arrow 2: Input Handler → API Layer
- Từ `36` (Video Input Handler) → `45/46` (REST API/WebSocket)
- **Purpose**: Truyền processed frames đến API để stream cho Frontend

### Arrow 3: Vector DB → API Layer  
- Từ `shared-db` → `45` (REST API)
- **Purpose**: Frontend query persons data qua REST API

### Arrow 4: Vector DB ↔ Cameras
- Từ `cam2-thanh-reid` (Body ReID) → `shared-db` (Vector DB)
- **Purpose**: Write body embeddings vào DB
- **Existing**: `arrow-huy-db` đã có cho Cam2 Huy module

## 📊 Current Backend Logic Flow:

```
Camera 4 Output (12)
    ↓ (Annotated Frame + Metadata)
Video Input Handler (13)
    ↓ (Query) ↔ (Store)
Vector Database (FAISS + Metadata)
    ↑ (Write embeddings from Camera 2 ReID)
Cam2 Thanh ReID (8)
    
API Layer (REST + WebSocket + Stream)
    ↓
Frontend Dashboard
```

## 🎯 Key Points:
1. **Bidirectional flow** giữa Input Handler và Vector DB là đúng
2. Camera modules write embeddings trực tiếp vào Vector DB
3. API Layer expose data cho Frontend
4. Tất cả components khác (Cameras 1-4, Future Enhancements, Performance) **giữ nguyên**
