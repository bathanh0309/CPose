"""
CPose — tools/run_phase2_analyzer.py
Standalone CLI to run Phase 2 (Offline Analyzer) on raw videos.
"""
import logging
import signal
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.analyzer import Analyzer
from app import get_config, RAW_VIDEOS_DIR, OUTPUT_DIR, MODEL_PHASE2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("[Tool-Phase2]")

def main():
    config = get_config()
    
    # Find all mp4 clips in raw_videos
    clips = list(RAW_VIDEOS_DIR.glob("*.mp4"))
    
    if not clips:
        logger.error("No clips found in %s", RAW_VIDEOS_DIR)
        return

    logger.info("Found %d clips for analysis.", len(clips))
    
    analyzer = Analyzer()
    
    # Start analysis
    analyzer.start(
        clips=clips,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PHASE2
    )

    # Signal handling for graceful stop
    def handle_sigint(sig, frame):
        logger.info("Stopping analyzer...")
        analyzer.stop()
        # Wait a bit for thread to finish
        for _ in range(5):
            if not analyzer.is_running():
                break
            time.sleep(1)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    logger.info("Analyzer running. Progress will be logged.")
    try:
        while analyzer.is_running():
            status = analyzer.status()
            if status.get("current_clip"):
                progress = status.get("progress_pct", 0)
                done = status.get("clips_done", 0)
                total = status.get("clips_total", 0)
                logger.info(f"Progress: {progress}% | Clips: {done}/{total} | Current: {status['current_clip']}")
            time.sleep(5)
            
    except KeyboardInterrupt:
        handle_sigint(None, None)

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
