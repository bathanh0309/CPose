from __future__ import annotations
import logging
import threading
from typing import Any
from app import socketio

logger = logging.getLogger("[WS]")
_pending: dict[str, dict[str, Any]] = {}


def emit_event_log(event: str, cam: str = "--"):
    payload = {
        "time": _now_time_string(),
        "event": event,
        "cam": cam,
    }
    socketio.emit("event_log", payload)


def emit_metric_log(
    *,
    cam: str = "--",
    fps: float | int | None = None,
    frame: str | int | None = None,
    conf: float | None = None,
    adl: str | None = None,
    event: str = "runtime",
):
    payload = {
        "time": _now_time_string(),
        "cam": cam,
        "fps": fps if fps is not None else "--",
        "frame": frame if frame is not None else "--",
        "conf": conf if conf is not None else "--",
        "adl": adl or "--",
        "event": event,
    }
    socketio.emit("metric_log", payload)


def emit_workspace_state(
    *,
    mode: str,
    running: bool,
    current_clip: str | None = None,
    current_cam: str | None = None,
    output_dir: str | None = None,
    queued: int = 0,
    staged_clips: list[str] | None = None,
    staged_camera_map: list[dict] | None = None,
):
    payload = {
        "time": _now_time_string(),
        "mode": mode,
        "running": running,
        "active_flow": mode if running and mode in {"rtsp", "multicam_folder"} else None,
        "current_clip": current_clip,
        "current_cam": current_cam,
        "output_dir": output_dir,
        "queued": queued,
    }
    if staged_clips is not None:
        payload["staged_clips"] = staged_clips
    if staged_camera_map is not None:
        payload["staged_camera_map"] = staged_camera_map
    socketio.emit("workspace_state", payload)


def emit_camera_status(cam_id: str, *, fps=None, frame=None, conf=None, status="IDLE"):
    socketio.emit(
        "camera_status",
        {
            "cam_id": cam_id,
            "fps": fps,
            "frame": frame,
            "conf": conf,
            "status": status,
        },
    )


def emit_rec_status(*, is_recording: bool, cam_id: str | None = None):
    socketio.emit(
        "rec_status",
        {
            "is_recording": is_recording,
            "cam_id": cam_id,
        },
    )


def emit_clip_saved(*, filename: str, cam_id: str, path: str, preview_url: str | None = None):
    socketio.emit(
        "clip_saved",
        {
            "filename": filename,
            "cam_id": cam_id,
            "path": path,
            "rel_path": path,
            "preview_url": preview_url,
        },
    )


def emit_pose_progress(
    *,
    cam_id: str,
    clip_stem: str,
    frame_id: int,
    total_frames: int,
    fps: float | None,
    conf: float | None,
    adl: str | None,
    pct: float | int | None,
    event: str = "pose_progress",
):
    socketio.emit(
        "pose_progress",
        {
            "cam_id": cam_id,
            "clip": clip_stem,
            "frame_id": frame_id,
            "total_frames": total_frames,
            "fps": fps,
            "conf": conf,
            "adl": adl,
            "pct": pct,
            "event": event,
        },
    )


def request_face_registration(clip_stem: str, cam_id: str, timeout: float | None = 300.0) -> dict | None:
    logger.info("Requesting face registration for %s (%s)", clip_stem, cam_id)

    ev = threading.Event()
    _pending[clip_stem] = {"event": ev, "payload": None}

    try:
        socketio.emit("register_face_request", {"clip_stem": clip_stem, "cam": cam_id})
    except Exception as exc:
        logger.warning("Failed to emit register_face_request: %s", exc)

    waited = ev.wait(timeout=timeout)
    entry = _pending.pop(clip_stem, None)
    if not waited or entry is None:
        logger.warning("Registration wait timed out or no response for %s", clip_stem)
        return None

    return entry.get("payload")


@socketio.on("connect")
def handle_connect():
    logger.info("UI connected via SocketIO")


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("UI disconnected from SocketIO")


@socketio.on("register_face_done")
def handle_register_face_done(data):
    clip_stem = str((data or {}).get("clip_stem", ""))
    if not clip_stem:
        logger.warning("register_face_done missing clip_stem")
        return

    entry = _pending.get(clip_stem)
    if not entry:
        logger.warning("No pending registration found for %s", clip_stem)
        return

    entry["payload"] = data
    entry["event"].set()


def _now_time_string() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%H:%M:%S")
