# CPose Src MLOps Pipeline

This document covers the terminal-only `src/` pipeline. It does not require the web app, Flask, Socket.IO, dashboard, or database.

## Run Pipeline

Default compatible command:

```bash
python -m src.pipeline.run_all --input data-test --output dataset/outputs
```

Full configured command on Windows:

```bat
python -m src.pipeline.run_all ^
  --input data-test ^
  --output dataset/outputs ^
  --manifest configs/multicam_manifest.example.json ^
  --topology configs/camera_topology.example.yaml ^
  --config configs/model_registry.demo_i5.yaml ^
  --gt dataset/annotations
```

Outputs are written to `dataset/outputs/pipeline/<timestamp>/` using the existing module folder convention: `1_detection`, `2_tracking`, `3_pose`, `4_adl`, optional `4b_face`, and `5_reid`.

## Evaluation

```bash
python -m src.evaluation.main --outputs dataset/outputs/pipeline/<timestamp> --gt dataset/annotations --out dataset/outputs/pipeline/<timestamp>/evaluation
```

Ground truth is optional. When GT files are absent, evaluation writes proxy metrics only and does not report accuracy.

## Benchmark

```bash
python -m src.pipeline.benchmark_all --run-dir dataset/outputs/pipeline/<timestamp>
```

Benchmark reports both `offline_module_sum_runtime_sec` and `pipeline_wall_clock_runtime_sec` when `pipeline_runtime.json` exists.

## Metric Types

- `proxy`: computed from predictions only, useful for debugging throughput and output quality, not paper accuracy.
- `ground_truth`: computed against annotation files under `dataset/annotations`.

Accuracy, macro-F1, global ID accuracy, and transfer success are only valid when the matching ground truth exists.

## Failure Reasons

Every module should write `failure_reason`. Values come from `src.common.errors.ErrorCode`, such as `LOW_KEYPOINT_VISIBILITY`, `UNCONFIRMED_TRACK`, `NO_FACE`, `TOPOLOGY_CONFLICT`, or `EVALUATION_SKIPPED_NO_GT`.

## Global ID vs Track ID

Local `track_id` is produced per video/camera by tracking. `global_id` is produced only by ReID and can persist across cameras. Do not treat local track IDs as global identities.

## Face Module

Face is optional and disabled in the demo config. If enabled without dependencies or models, it still writes `face_events.json` and marks missing results with `MODEL_MISSING` or `NO_FACE` rather than fake embeddings or anti-spoof decisions.

## Remaining Paper TODOs

- Add complete MOT IDF1/HOTA metrics for tracking.
- Add validated face detector/recognizer and anti-spoof models.
- Add richer ReID GT annotations for false split/merge and blind-zone recovery.
- Validate topology polygons per deployment camera geometry.
- Calibrate ADL thresholds per FPS/resolution/camera angle.
