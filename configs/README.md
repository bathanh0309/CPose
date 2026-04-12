# Shared Configs

Use this folder as the source of truth for project-wide configuration.

Current shared configs:
- `pose_adl.yaml`: Phase 2 pose + ADL pipeline config.
- `runtime.env`: optional centralized runtime env for Phase 1 backend.

Compatibility rules:
- `feat-pose-adl/backend/src/config.yaml` is only a compatibility copy.
- `feat-realtime-data/backend/app/config.py` prefers `configs/runtime.env` and falls back to root `.env`.
