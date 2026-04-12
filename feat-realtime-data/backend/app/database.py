# Tầng truy cập SQLite để lưu sự kiện và cấu hình camera riêng tư.
"""
SQLite persistence layer for event logs and private camera configuration.
Uses aiosqlite for non-blocking I/O compatible with FastAPI's asyncio loop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiosqlite

from app.config import get_settings
from app.schemas import EventCreate, EventOut, EventType

_settings = get_settings()
DB_PATH = str(_settings.db_path)


CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    time        TEXT NOT NULL,
    event       TEXT NOT NULL,
    person      TEXT NOT NULL DEFAULT 'Unknown',
    cam         TEXT NOT NULL,
    camera_id   TEXT NOT NULL,
    clip_path   TEXT
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_events_time ON events (time DESC);
"""

CREATE_CAMERA_RESOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS camera_resources (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    file_name   TEXT NOT NULL,
    content     TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class CameraResourcesRecord:
    file_name: str
    content: str
    uploaded_at: str


async def init_db() -> None:
    """Create tables on startup."""
    _settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_EVENTS_TABLE)
        await db.execute(CREATE_INDEX)
        await db.execute(CREATE_CAMERA_RESOURCES_TABLE)
        await db.commit()


async def insert_event(evt: EventCreate) -> EventOut:
    """Persist a new event and return its full representation."""
    event_id = str(uuid.uuid4())
    time_str = evt.time.isoformat() + "Z"

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO events (id, time, event, person, cam, camera_id, clip_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                time_str,
                evt.event.value,
                evt.person,
                evt.cam_name,
                evt.camera_id,
                evt.clip_path,
            ),
        )
        await db.commit()

    return EventOut(
        id=event_id,
        time=time_str,
        event=evt.event,
        person=evt.person,
        cam=evt.cam_name,
        camera_id=evt.camera_id,
        clip_path=evt.clip_path,
    )


async def fetch_recent_events(limit: int = 100) -> list[EventOut]:
    """Return the N most recent events, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM events ORDER BY time DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()

    return [
        EventOut(
            id=row["id"],
            time=row["time"],
            event=EventType(row["event"]),
            person=row["person"],
            cam=row["cam"],
            camera_id=row["camera_id"],
            clip_path=row["clip_path"],
        )
        for row in rows
    ]


async def save_camera_resources(file_name: str, content: str) -> CameraResourcesRecord:
    """Persist the active camera resources payload in a private backend table."""
    uploaded_at = datetime.utcnow().isoformat() + "Z"

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO camera_resources (id, file_name, content, uploaded_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                file_name = excluded.file_name,
                content = excluded.content,
                uploaded_at = excluded.uploaded_at
            """,
            (file_name, content, uploaded_at),
        )
        await db.commit()

    return CameraResourcesRecord(
        file_name=file_name,
        content=content,
        uploaded_at=uploaded_at,
    )


async def fetch_camera_resources() -> Optional[CameraResourcesRecord]:
    """Return the currently stored private camera resources payload, if any."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT file_name, content, uploaded_at
            FROM camera_resources
            WHERE id = 1
            """
        )
        row = await cursor.fetchone()

    if row is None:
        return None

    return CameraResourcesRecord(
        file_name=row["file_name"],
        content=row["content"],
        uploaded_at=row["uploaded_at"],
    )
