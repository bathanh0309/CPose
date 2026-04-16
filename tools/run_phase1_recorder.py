import argparse
import logging
import time
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.recorder import RecorderManager

def main():
    parser = argparse.ArgumentParser(description="Phase 1: RTSP Recorder CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--output-dir", type=str, default="data/raw_videos", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("[RecorderCLI]")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    recorder = RecorderManager()
    
    # Mock socketio for CLI
    class MockSocketIO:
        def emit(self, event, data):
            if event == "rec_log":
                logger.info(f"[{data.get('cam_id')}] {data.get('message')}")
            elif event == "clip_saved":
                logger.info(f"[{data.get('cam_id')}] clip_saved: {data.get('filename')}")

    import app
    app.socketio = MockSocketIO()

    storage_limit = float(config.get("phase1", {}).get("storage_limit_gb", 10.0))
    model_path = Path(config.get("phase1", {}).get("model", "models/yolo11n.pt"))
    cameras = config.get("cameras", [])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Phase 1 Recorder for {len(cameras)} cameras...")
    recorder.start(
        cameras=cameras,
        storage_limit_gb=storage_limit,
        output_dir=out_dir,
        model_path=model_path,
        config=config.get("phase1", {})
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping recorder...")
        recorder.stop()

if __name__ == "__main__":
    main()
