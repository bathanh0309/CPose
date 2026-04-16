import argparse
import logging
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.engines.phase3 import run_phase3
from app.core.adl_model import ADLModelWrapper
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Pose + ADL Recognizer")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing MP4s")
    parser.add_argument("--output-dir", type=str, default="data/output_pose", help="Output directory")
    parser.add_argument("--pose-model", type=str, required=True, help="YOLO Pose model path")
    parser.add_argument("--adl-mode", type=str, choices=["rule", "gcn"], default="rule", help="ADL recognition mode")
    parser.add_argument("--adl-cfg", type=str, help="GCN config path (required for gcn mode)")
    parser.add_argument("--adl-ckpt", type=str, help="GCN checkpoint path (required for gcn mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("[RecognizerCLI]")

    # Load Pose Model
    logger.info(f"Loading Pose model: {args.pose_model}")
    model = YOLO(args.pose_model)

    # Load ADL Model if GCN
    adl_model = None
    if args.adl_mode == "gcn":
        if not args.adl_cfg or not args.adl_ckpt:
            logger.error("GCN mode requires --adl-cfg and --adl-ckpt")
            sys.exit(1)
        logger.info(f"Loading ADL GCN Model: {args.adl_ckpt}")
        adl_model = ADLModelWrapper(cfg_path=args.adl_cfg, ckpt_path=args.adl_ckpt)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    clips = list(input_path.glob("**/*.mp4"))
    
    if not clips:
        logger.error(f"No clips found in {args.input_dir}")
        return

    logger.info(f"Processing {len(clips)} clips...")
    
    for clip in clips:
        logger.info(f"--- Processing {clip.name} ---")
        try:
            # Simple fallback config
            config = {
                "window_size": 30,
                "keypoint_conf_min": 0.3,
                "conf_threshold": 0.45,
                "person_class_id": 0
            }
            
            result = run_phase3(
                model=model,
                adl_model=adl_model,
                clip_path=clip,
                output_dir=output_path,
                config=config,
                save_overlay=True
            )
            logger.info(f"Done: {clip.name}. Keypoints: {result['keypoints_written']}, ADLs: {result['adl_events']}")
        except Exception as e:
            logger.error(f"Failed to process {clip.name}: {e}")

if __name__ == "__main__":
    main()
