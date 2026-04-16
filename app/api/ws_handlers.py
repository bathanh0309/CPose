"""WebSocket handlers and helper for requesting face registration from UI.

This module uses Flask-SocketIO (already configured in `app.__init__`).
Recognizer threads can call `request_face_registration(clip_stem, cam_id)`
to emit a `register_face_request` to the frontend and block until the
frontend emits back `register_face_done` with payload.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app import socketio

logger = logging.getLogger("[WS]")

import asyncio
from app.utils.serialize import track_to_json
from cpose.pipeline.multicam_online_system import MultiCamOnlineSystem

# pending map: clip_stem -> {event: threading.Event, payload: Any}
_pending: dict[str, dict[str, Any]] = {}

def push_camera_update(tracks: list, timestamp: float | None = None):
    # Existing helper
    pass

async def camera_stream_ws(ws, system: MultiCamOnlineSystem):
    """Loop to push snapshot updates to the UI."""
    while True:
        ts, tracks = system.get_current_tracks()
        payload = {
            "type": "camera_update",
            "timestamp": ts,
            "tracks": [track_to_json(t) for t in tracks],
        }
        try:
            # Note: flask-socketio uses emit, but for pure websocket snippets:
            # await ws.send_json(payload)
            socketio.emit("camera_update", payload)
        except Exception:
            break
        await asyncio.sleep(0.1)
    """Broadcast current track states to all connected UIs."""
    if timestamp is None:
        import time
        timestamp = time.time()
        
    payload = {
        "type": "camera_update",
        "timestamp": timestamp,
        "tracks": [track_to_json(t) for t in tracks],
    }
    try:
        socketio.emit("camera_update", payload)
    except Exception as exc:
        logger.warning("Failed to emit camera_update: %s", exc)


def request_face_registration(clip_stem: str, cam_id: str, timeout: float | None = 300.0) -> dict | None:
    """Emit a register request and wait for the UI response.

    Returns the payload from the client or None on timeout / disconnect.
    """
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

    logger.info("Received registration for %s", clip_stem)
    return entry.get("payload")


@socketio.on("connect")
def _handle_connect():
    logger.info("UI connected via SocketIO")


@socketio.on("disconnect")
def _handle_disconnect():
    logger.info("UI disconnected from SocketIO")


@socketio.on("register_face_done")
def _handle_register_face_done(data):
    """Client sends back registration payload.

    Expected data: {"clip_stem": "...", "name": "..", "age":.., "person_id": ".."}
    """
    try:
        clip_stem = str(data.get("clip_stem", ""))
    except Exception:
        clip_stem = ""

    if not clip_stem:
        logger.warning("register_face_done missing clip_stem: %s", data)
        return

    entry = _pending.get(clip_stem)
    if not entry:
        logger.warning("No pending registration found for %s", clip_stem)
        return

    entry["payload"] = data
    entry["event"].set()
    logger.info("Set pending registration result for %s", clip_stem)
