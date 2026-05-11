# CPose Configs

Config is loaded from one base tree plus one profile override. Runtime secrets stay in `.env` and are not committed.

```text
configs/
  base/                 default values shared by every environment
  camera/               topology and manifest without credentials
  profiles/             environment or hardware overrides
  phase.yaml            current run phase settings
  unified_config.yaml   app-level source of truth for the Flask app
  .env.example          credential template
```

## Runtime Profiles

Use `src.config.load_config(profile="dev")` or pass one of these files to CLI `--config` / `--models`:

| Profile | Purpose |
|---|---|
| `profiles/dev.yaml` | Local laptop/dev defaults |
| `profiles/edge.yaml` | CPU/edge constrained defaults |
| `profiles/benchmark.yaml` | Full metrics and comparison output |

## Public vs Private

Commit `configs/.env.example`, never commit `.env` or `configs/_private.yaml`.
RTSP URLs and credentials should be provided through environment variables such as `RTSP_CAM1` and `RTSP_CAM2`.
