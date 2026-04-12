from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from threading import Event

from app.services.processing_service import processing_service

logger = logging.getLogger("phase2_runner")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_POSE_ADL_SRC_DIR = _REPO_ROOT / "feat-pose-adl" / "backend" / "src"
if str(_POSE_ADL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_POSE_ADL_SRC_DIR))

from pose_adl_processor import PoseAdlClipProcessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 worker: process queued raw RTSP clips with pose + ADL.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds to wait before checking the queue again when idle.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one queued clip, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    stop_event = Event()

    def _handle_stop(_sig, _frame) -> None:
        logger.info("Stop signal received. Phase 2 worker is shutting down.")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    processing_service.initialize()
    processor = PoseAdlClipProcessor()

    logger.info("Phase 2 worker ready")
    logger.info("Raw queue: %s", processing_service.snapshot()["raw_videos_dir"])
    logger.info("Processed output: %s", processing_service.snapshot()["processed_videos_dir"])

    processed_any = False
    while not stop_event.is_set():
        job = processing_service.claim_next_job()
        if job is None:
            if args.once:
                break
            time.sleep(max(args.poll_interval, 0.5))
            continue

        processed_any = True
        raw_path = Path(job["raw_path"])
        processed_path = Path(job["processed_path"])
        logger.info("Processing %s", raw_path.name)
        try:
            processor.process_video(
                input_path=raw_path,
                output_path=processed_path,
                stop_event=stop_event,
            )
        except Exception as exc:
            if stop_event.is_set():
                processing_service.requeue(raw_path)
                logger.warning("Interrupted while processing %s; job re-queued", raw_path.name)
                break

            processing_service.mark_failed(raw_path, str(exc))
            logger.exception("Phase 2 failed for %s", raw_path.name)
            if args.once:
                return 1
            continue

        processing_service.mark_completed(raw_path)
        logger.info("Completed %s -> %s", raw_path.name, processed_path.name)
        if args.once:
            break

    if args.once and not processed_any:
        logger.info("No queued jobs found.")

    processing_service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
