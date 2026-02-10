# 🚀 HAVEN Production Refactor - Quick Start Guide

## 📁 Cấu trúc thư mục

```
D:\HAVEN\backend\
├── step1.bat              # Phase 1: Infrastructure Testing
├── step2.bat              # Phase 2: Deep ReID Engine (Optional)
├── step3.bat              # Phase 3: Master-Slave Logic
├── step4.bat              # Phase 4: Dangerous Zones Setup
├── step5.bat              # Phase 5: Pipeline Integration
├── step6.bat              # Phase 6: Tuning & Optimization
├── run_all_phases.bat     # Chạy tất cả các phase
│
├── core/                  # ✅ Core logic modules
│   └── global_id_manager.py
├── storage/               # ✅ Persistence & Vector DB
│   ├── persistence.py
│   └── vector_db.py
├── tests/                 # ✅ Unit tests
│   └── test_global_id_manager.py
├── config/                # ✅ Configuration
│   └── production.yaml
│
└── [Pending Implementation]
    ├── models/            # Phase 2: OSNet wrapper
    ├── modules/           # Phase 4: Dangerous zones/objects
    ├── pipeline/          # Phase 5: Multi-camera pipeline
    ├── utils/             # Phase 5: Utilities
    └── scripts/           # Phase 5: Run scripts
```

---

## 🎯 Cách sử dụng

### Option 1: Chạy từng phase (Khuyến nghị)

Chạy từng bước để kiểm tra kỹ:

```batch
# Phase 1: Test infrastructure (REQUIRED)
D:\HAVEN\backend> .\step1.bat

# Phase 2: OSNet setup (OPTIONAL - cần C++ compiler)
D:\HAVEN\backend> .\step2.bat

# Phase 3: Test master-slave logic (REQUIRED)
D:\HAVEN\backend> .\step3.bat

# Phase 4: Setup dangerous zones (OPTIONAL)
D:\HAVEN\backend> .\step4.bat

# Phase 5: Setup pipeline structure (OPTIONAL)
D:\HAVEN\backend> .\step5.bat

# Phase 6: Final validation (REQUIRED)
D:\HAVEN\backend> .\step6.bat
```

### Option 2: Chạy tất cả cùng lúc

```batch
D:\HAVEN\backend> .\run_all_phases.bat
```

Script sẽ tự động chạy tất cả các phase và hỏi bạn có muốn tiếp tục không.

---

## ✅ Phase Status

| Phase | Status | Description | Required? |
|-------|--------|-------------|-----------|
| **Phase 1** | ✅ **COMPLETE** | Infrastructure (Persistence, Vector DB, GlobalIDManager) | **YES** |
| **Phase 2** | ⚠️ Optional | Deep ReID Engine (OSNet) - requires C++ compiler | NO |
| **Phase 3** | ✅ **COMPLETE** | Master-Slave Logic - All tests passing | **YES** |
| **Phase 4** | 📋 Pending | Dangerous Zones & Objects - Structure only | NO |
| **Phase 5** | 📋 Pending | Pipeline Integration - Structure only | NO |
| **Phase 6** | ✅ Ready | Tuning & Optimization - Can run anytime | **YES** |

---

## 🧪 Test Results Summary

### Phase 1 Tests (Infrastructure)
```
✅ test_creates_first_global_id          # Master creates G1
✅ test_creates_sequential_ids           # Master creates G1-G10
✅ test_matches_existing_person          # Matching works
✅ test_temporal_voting_prevents_flicker # No ID jitter
✅ test_never_creates_global_id          # Slave never creates
✅ test_matches_master_created_id        # Slave matches G1
✅ test_assigns_unk_when_no_match        # UNK assignment
✅ test_unk_resurrection_via_iou         # UNK resurrection
✅ test_optimal_assignment_two_people    # Hungarian works
✅ test_restart_recovery                 # Persistence works
✅ test_quality_gating                   # Quality filtering
✅ test_unknown_camera                   # Error handling
```

**Result:** 12/12 tests PASSED ✅

---

## 📊 Current System Capabilities

### ✅ Working Features
1. **Persistence Layer**
   - SQLite database for metadata
   - Memory-mapped embeddings (10,000 capacity)
   - Atomic writes with WAL mode
   - Restart recovery

2. **Vector Database**
   - FAISS IndexFlatIP for cosine similarity
   - Fallback to linear search if FAISS unavailable
   - Auto-upgrade strategy (Flat → HNSW → IVF)

3. **GlobalIDManager**
   - Master camera creates G1, G2, G3...
   - Slave cameras NEVER create GlobalIDs
   - Temporal voting (3 frames confirmation)
   - UNK resurrection via IoU
   - Hungarian assignment ready

### ⚠️ Pending Implementation
1. **OSNet ReID** (Phase 2)
   - Requires: Microsoft Visual C++ Build Tools
   - Status: Optional, system works without it
   - Fallback: Heuristic features available

2. **Dangerous Zones** (Phase 4)
   - Point-in-polygon detection
   - Dwell time monitoring
   - Alert cooldown

3. **Pipeline Integration** (Phase 5)
   - Multi-camera stream manager
   - Main processor
   - Visualization

---

## 🔧 Troubleshooting

### Issue: Tests fail on Windows
**Solution:** Tests are now Windows-compatible with proper file handle cleanup.

### Issue: OSNet installation fails (Phase 2)
**Solution:** 
1. Install Microsoft Visual C++ Build Tools
2. OR skip Phase 2 - system works without OSNet
3. Heuristic features will be used as fallback

### Issue: Permission errors when deleting temp files
**Solution:** Already fixed with `gc.collect()` in persistence layer.

### Issue: FAISS not available
**Solution:** System automatically falls back to linear search.

---

## 📖 Documentation

- **README_PRODUCTION.md** - Production deployment guide
- **IMPLEMENTATION_ROADMAP.md** - Detailed implementation plan
- **DELIVERABLES_SUMMARY.md** - Project summary and deliverables
- **config/production.yaml** - System configuration

---

## 🎓 Next Steps

### For Development:
1. ✅ Run `step1.bat` to verify infrastructure
2. ⚠️ (Optional) Run `step2.bat` to install OSNet
3. ✅ Run `step3.bat` to verify master-slave logic
4. 📋 Implement Phase 4 modules (dangerous zones)
5. 📋 Implement Phase 5 pipeline
6. ✅ Run `step6.bat` for final validation

### For Production:
1. Review and tune `config/production.yaml`
2. Set up data directories
3. Configure camera sources
4. Run system: `python scripts/run_multi_camera.py`

---

**Last Updated:** 2026-02-02  
**Version:** 2.0  
**Status:** Core Infrastructure Complete ✅
