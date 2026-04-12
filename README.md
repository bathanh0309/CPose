# CPose

CPose is a local Flask dashboard for a three-phase person-monitoring workflow:

1. Phase 1 reads RTSP streams and saves short MP4 clips when a person is detected.
2. Phase 2 scans saved clips offline and exports PNG frames plus bounding-box labels.
3. Phase 3 runs pose estimation offline and exports keypoints plus rule-based ADL labels.

## Canonical structure

```text
Capstone_Project/
|-- app/
|-- configs/
|-- data/
|   |-- config/
|   |-- raw_videos/
|   |-- output_labels/
|   `-- output_pose/
|-- models/
|-- static/
|-- main.py
|-- run.bat
|-- requirements.txt
|-- CLAUDE.md
`-- README.md
```

Legacy feature branches and duplicate backends are intentionally removed from the main runtime path. The Flask app under `app/` is the single source of truth.

## Run

```bat
run.bat
```

Or manually:

```bash
pip install -r requirements.txt
python main.py
```

Open `http://localhost:5000`.
