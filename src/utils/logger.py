import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_frame_metrics(
    logger: logging.Logger,
    module: str,
    camera_id: str,
    frame_idx: int,
    fps: float,
    interval: int = 1,
    **metrics,
):
    interval = max(1, int(interval))
    if int(frame_idx) % interval != 0:
        return
    metric_text = " ".join(f"{key}={value}" for key, value in metrics.items())
    logger.info(
        f"[METRIC] module={module} camera={camera_id} frame={int(frame_idx)} fps={float(fps):.1f}"
        + (f" {metric_text}" if metric_text else "")
    )
