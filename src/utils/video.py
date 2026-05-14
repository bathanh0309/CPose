from pathlib import Path

import cv2

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")


def find_default_video_source(root: Path):
    root = Path(root)
    sample = root / "data" / "sample.mp4"
    if sample.exists():
        return str(sample)

    input_dir = root / "data" / "input"
    if input_dir.exists():
        videos = sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        )
        if videos:
            return str(videos[0])

    return None


def parse_video_source(source: str):
    source = str(source)
    return int(source) if source.isdigit() else source


def open_video_source(source: str):
    parsed = parse_video_source(source)
    cap = cv2.VideoCapture(parsed)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap, parsed


def get_video_meta(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return width, height, fps, total_frames


def create_video_writer(output_path, fps, width, height):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")
    return writer


def safe_imshow(window_name, frame, delay=1):
    cv2.imshow(window_name, frame)
    return cv2.waitKey(delay) & 0xFF
