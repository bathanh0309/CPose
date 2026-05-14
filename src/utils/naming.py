from pathlib import Path
import re


def sanitize_token(value: object, default: str = "x") -> str:
    """Convert any value into a compact filesystem-safe token."""
    text = str(value).strip()
    if not text:
        return default

    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def short_camera_id(camera_id: object) -> str:
    """
    Convert camera id to compact format.

    Examples:
        cam01 -> c01
        cam02 -> c02
        camera_2 -> c02
        2 -> c02
    """
    text = sanitize_token(camera_id, "cam")
    numbers = re.findall(r"\d+", text)

    if numbers:
        return f"c{int(numbers[-1]):02d}"

    if text.lower().startswith("cam"):
        return "c" + text[3:]

    return text.lower()[:8]


def compact_global_id(global_id: object) -> str:
    """
    Convert global id to compact format.

    Examples:
        gid_00001 -> g001
        APhu -> APhu
        unknown -> unk
    """
    text = sanitize_token(global_id, "unk")

    if text.lower() in {"unknown", "unk"}:
        return "unk"

    numbers = re.findall(r"\d+", text)
    if text.lower().startswith("gid") and numbers:
        return f"g{int(numbers[-1]):03d}"

    return text[:12]


def make_clip_id(
    camera_id: object,
    local_track_id: int,
    global_id: object,
    clip_index: int,
) -> str:
    """
    Compact clip id for ADL pkl.

    Example:
        c02_t002_g001_001
    """
    cam = short_camera_id(camera_id)
    gid = compact_global_id(global_id)
    return f"{cam}_t{int(local_track_id):03d}_{gid}_{int(clip_index):03d}"


def make_video_output_name(module: str, camera_id: object, ext: str = ".mp4") -> str:
    """
    Compact output video name.

    Examples:
        pose_cam02.mp4
        reid_cam02.mp4
        adl_cam02.mp4
        pipe_cam02.mp4
    """
    module = sanitize_token(module, "out").lower()

    aliases = {
        "pipeline": "pipe",
        "full": "pipe",
        "tracking": "track",
    }
    module = aliases.get(module, module)

    cam = short_camera_id(camera_id).replace("c", "cam", 1)

    if not ext.startswith("."):
        ext = "." + ext

    return f"{module}_{cam}{ext}"


def make_json_output_name(module: str, camera_id: object, ext: str = ".json") -> str:
    module = sanitize_token(module, "out").lower()
    cam = short_camera_id(camera_id).replace("c", "cam", 1)

    if not ext.startswith("."):
        ext = "." + ext

    return f"{module}_{cam}{ext}"


def resolve_output_path(output_dir, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename
