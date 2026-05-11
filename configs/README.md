# CPose Configs

The current CLI pipeline uses only these files by default:

| File | Used by | Purpose |
|---|---|---|
| `model_registry.demo_i5.yaml` | Modules 1-6 | Model paths, thresholds, and light runtime defaults for local demos |
| `camera_topology.yaml` | Module 5 and live pipeline | Allowed camera transitions for cross-camera ReID |
| `multicam_manifest.json` | Module 5 | Optional camera ID and start-time metadata for each video |

Other YAML files in this folder are legacy/reference configs kept for older app code.
Do not edit them for the normal `run_01_*.bat` to `run_07_*.bat` workflow unless that older code is being used.
