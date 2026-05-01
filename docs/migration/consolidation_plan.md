# Project Consolidation Plan

## Phase 1: Recorder Merge (`data-collect-thanh` -> `app/services/recorder.py`)
- [ ] Refactor `CameraWorker` in `recorder.py` to use multi-threading (Ingest, Detect, Preview) for better performance.
- [ ] Implement robust reconnection with exponential backoff (from `data-collect-thanh`).
- [ ] Integrate separate queues for detection and preview.
- [ ] Ensure config parameters are pulled from `configs/config.yaml`.

## Phase 2: Analyzer Merge (`data-labeling` -> `app/services/analyzer.py`)
- [ ] Update `Analyzer` to support `FRAME_SKIP` to speed up offline labeling.
- [ ] Standardize output format for PNG frames and `detections.txt` based on `data-labeling` conventions.
- [ ] Implement stop-flag/cancellation support for long-running analysis jobs.
- [ ] Add progress reporting via Socket.IO.

## Phase 3: Tools & Config
- [ ] Create `tools/run_phase1_recorder.py`.
- [ ] Create `tools/run_phase2_analyzer.py`.
- [ ] Consolidate camera URLs into `data/config/resources.txt`.
- [ ] Centralize thresholds and tuning in `configs/config.yaml`.


- [ ] Update `README.md` and `docs/migration/` notes.
