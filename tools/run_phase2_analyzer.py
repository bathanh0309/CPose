import argparse
import logging
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.analyzer import Analyzer

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Offline Analyzer")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing MP4s")
    parser.add_argument("--output-dir", type=str, default="data/output_labels", help="Output directory")
    parser.add_argument("--model-path", type=str, default="models/yolo11n.pt", help="YOLO model path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("[AnalyzerCLI]")

    # Mock socketio
    class MockSocketIO:
        def emit(self, event, data):
            if event == "analysis_progress":
                print(f"\rProgress: {data.get('pct')}% | Clip: {data.get('clip')} | Saved: {data.get('frames_saved')}", end="")
            elif event == "analysis_complete":
                print(f"\nAnalysis complete. Clips: {data.get('clips_done')}, Frames: {data.get('frames_saved')}")

    import app
    app.socketio = MockSocketIO()

    analyzer = Analyzer()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    model_path = Path(args.model_path)
    
    clips = list(input_path.glob("**/*.mp4"))
    if not clips:
        logger.error(f"No MP4 clips found in {args.input_dir}")
        return

    logger.info(f"Analyzing {len(clips)} clips from {args.input_dir}...")
    analyzer.start(clips=clips, output_dir=output_path, model_path=model_path)

    # Wait for completion
    while analyzer.is_running():
        import time
        time.sleep(0.5)

if __name__ == "__main__":
    main()
