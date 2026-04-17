"""
CPose — tools/run_phase1_recorder.py
Standalone CLI to run Phase 1 (Recording) based on data/config/resources.txt
"""
import logging
import signal
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.recorder import RecorderManager
from app import get_config, RAW_VIDEOS_DIR, MODEL_PHASE1, RESOURCES_FILE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("[Tool-Phase1]")

def parse_cameras(file_path: Path):
    cameras = []
    if not file_path.exists():
        return cameras
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "__" in line:
                label, url = line.split("__", 1)
                cameras.append({
                    "cam_id": len(cameras) + 1,
                    "label": label.strip(),
                    "url": url.strip()
                })
            else:
                cameras.append({
                    "cam_id": len(cameras) + 1,
                    "label": f"cam{len(cameras)+1}",
                    "url": line
                })
    return cameras

def main():
    config = get_config()
    cameras = parse_cameras(RESOURCES_FILE)
    
    if not cameras:
        logger.error("No cameras found in %s", RESOURCES_FILE)
        return

    logger.info("Starting Phase 1 Recorder for %d cameras...", len(cameras))
    
    recorder = RecorderManager()
    
    # Mock emitter for CLI
    def _console_emit(event, data):
        if event == "rec_log":
            logger.info("REC [%s] %s", data.get("cam_id"), data.get("message"))
        elif event == "detection_event":
             logger.info("DET [%s] Person detected!", data.get("cam_id"))

    # We need to monkey-patch RecorderManager.start slightly or provide a custom start
    # Since we modified recorder.py to be more robust, let's use it.
    
    recorder.start(
        cameras=cameras,
        storage_limit_gb=config.phase1.storage_limit_gb,
        output_dir=RAW_VIDEOS_DIR,
        model_path=MODEL_PHASE1,
        config=config.phase1,
        # We'll pass a custom emitter here if we were to refactor RecorderManager.start
        # but for now, it's tied to 'app.socketio'. 
        # For a true standalone tool, we might want to decouple it further.
    )

    # Signal handling for graceful stop
    def handle_sigint(sig, frame):
        logger.info("Stopping recorder...")
        recorder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    logger.info("Recorder running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handle_sigint(None, None)

if __name__ == "__main__":
    main()
