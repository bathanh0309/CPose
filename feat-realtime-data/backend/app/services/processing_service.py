# Shared Phase 2 queue/state service backed by SQLite so Phase 1 and Phase 2
# can run as separate processes while using the same job registry.
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.services.file_namer import processed_clip_path

logger = logging.getLogger(__name__)
_settings = get_settings()
_STALE_PROCESSING_SECS = 15 * 60

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processing_jobs (
    raw_path             TEXT PRIMARY KEY,
    raw_file_name        TEXT NOT NULL,
    camera_id            TEXT,
    camera_name          TEXT,
    status               TEXT NOT NULL,
    processed_path       TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    started_at           TEXT,
    completed_at         TEXT,
    error                TEXT
);
"""

_CREATE_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status
ON processing_jobs (status, created_at);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


class ProcessingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self._ensure_storage()
        self.synchronize()

    def shutdown(self) -> None:
        return

    def enqueue_clip(
        self,
        raw_path: Path | str,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
    ) -> None:
        self._ensure_storage()

        raw_file = Path(raw_path).resolve()
        processed_file = processed_clip_path(raw_file, _settings.processed_videos_dir)
        created_at = _utc_from_ts(raw_file.stat().st_mtime) if raw_file.exists() else _utc_now()
        updated_at = _utc_now()
        status = "completed" if processed_file.exists() else "queued"
        completed_at = _utc_from_ts(processed_file.stat().st_mtime) if processed_file.exists() else None

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT camera_id, camera_name FROM processing_jobs WHERE raw_path = ?",
                (str(raw_file),),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    raw_path, raw_file_name, camera_id, camera_name, status,
                    processed_path, created_at, updated_at, started_at,
                    completed_at, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)
                ON CONFLICT(raw_path) DO UPDATE SET
                    raw_file_name = excluded.raw_file_name,
                    camera_id = COALESCE(excluded.camera_id, processing_jobs.camera_id),
                    camera_name = COALESCE(excluded.camera_name, processing_jobs.camera_name),
                    status = CASE
                        WHEN excluded.status = 'completed' THEN 'completed'
                        WHEN processing_jobs.status = 'processing' THEN 'processing'
                        ELSE 'queued'
                    END,
                    processed_path = excluded.processed_path,
                    updated_at = excluded.updated_at,
                    completed_at = CASE
                        WHEN excluded.status = 'completed' THEN excluded.completed_at
                        ELSE processing_jobs.completed_at
                    END,
                    error = CASE
                        WHEN excluded.status = 'completed' THEN NULL
                        WHEN processing_jobs.status = 'processing' THEN processing_jobs.error
                        ELSE NULL
                    END
                """,
                (
                    str(raw_file),
                    raw_file.name,
                    camera_id or (existing["camera_id"] if existing else None),
                    camera_name or (existing["camera_name"] if existing else None),
                    status,
                    str(processed_file),
                    created_at,
                    updated_at,
                    completed_at,
                ),
            )
            conn.commit()

    def retry(self, raw_file_name: str) -> bool:
        self._ensure_storage()
        self.synchronize()

        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT raw_path
                FROM processing_jobs
                WHERE raw_file_name = ? AND status = 'failed'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (raw_file_name,),
            ).fetchone()
            if row is None:
                return False

            conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'queued',
                    updated_at = ?,
                    started_at = NULL,
                    completed_at = NULL,
                    error = NULL
                WHERE raw_path = ?
                """,
                (_utc_now(), row["raw_path"]),
            )
            conn.commit()
        return True

    def snapshot(self) -> dict:
        self._ensure_storage()
        self.synchronize()

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM processing_jobs
                ORDER BY created_at DESC
                """
            ).fetchall()

        jobs = [self._serialize_row(row) for row in rows if Path(row["raw_path"]).exists()]
        return {
            "raw_videos_dir": str(_settings.recordings_dir),
            "processed_videos_dir": str(_settings.processed_videos_dir),
            "queued": sum(1 for job in jobs if job["status"] == "queued"),
            "processing": sum(1 for job in jobs if job["status"] == "processing"),
            "completed": sum(1 for job in jobs if job["status"] == "completed"),
            "failed": sum(1 for job in jobs if job["status"] == "failed"),
            "jobs": jobs,
        }

    def claim_next_job(self) -> Optional[dict]:
        self._ensure_storage()
        self.synchronize()

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT raw_path, raw_file_name, processed_path, camera_id, camera_name
                FROM processing_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                conn.commit()
                return None

            now = _utc_now()
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'processing',
                    started_at = ?,
                    updated_at = ?,
                    error = NULL
                WHERE raw_path = ?
                """,
                (now, now, row["raw_path"]),
            )
            conn.commit()

        return dict(row)

    def mark_completed(self, raw_path: Path | str) -> None:
        self._set_terminal_state(Path(raw_path), status="completed", error=None)

    def mark_failed(self, raw_path: Path | str, error: str) -> None:
        self._set_terminal_state(Path(raw_path), status="failed", error=error)

    def requeue(self, raw_path: Path | str) -> None:
        raw_file = Path(raw_path).resolve()
        self._ensure_storage()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'queued',
                    updated_at = ?,
                    started_at = NULL,
                    completed_at = NULL,
                    error = NULL
                WHERE raw_path = ?
                """,
                (_utc_now(), str(raw_file)),
            )
            conn.commit()

    def synchronize(self) -> None:
        self._ensure_storage()

        raw_files = {
            str(raw_file.resolve()): raw_file.resolve()
            for raw_file in sorted(_settings.recordings_dir.glob("*.mp4"))
        }
        now = _utc_now()
        stale_before = datetime.now(timezone.utc).timestamp() - _STALE_PROCESSING_SECS

        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM processing_jobs").fetchall()
            rows_by_path = {row["raw_path"]: row for row in rows}

            for raw_path_str, raw_file in raw_files.items():
                processed_file = processed_clip_path(raw_file, _settings.processed_videos_dir)
                row = rows_by_path.get(raw_path_str)
                created_at = _utc_from_ts(raw_file.stat().st_mtime)

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO processing_jobs (
                            raw_path, raw_file_name, camera_id, camera_name, status,
                            processed_path, created_at, updated_at, started_at,
                            completed_at, error
                        )
                        VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, NULL, ?, NULL)
                        """,
                        (
                            raw_path_str,
                            raw_file.name,
                            "completed" if processed_file.exists() else "queued",
                            str(processed_file),
                            created_at,
                            now,
                            _utc_from_ts(processed_file.stat().st_mtime) if processed_file.exists() else None,
                        ),
                    )
                    continue

                status = row["status"]
                error = row["error"]
                started_at = row["started_at"]
                completed_at = row["completed_at"]

                if processed_file.exists():
                    status = "completed"
                    completed_at = _utc_from_ts(processed_file.stat().st_mtime)
                    error = None
                elif status == "completed":
                    status = "queued"
                    completed_at = None
                elif status == "processing":
                    started_dt = _parse_iso(started_at)
                    if started_dt is None or started_dt.timestamp() < stale_before:
                        status = "queued"
                        started_at = None
                        error = "Re-queued after stale Phase 2 worker state."

                conn.execute(
                    """
                    UPDATE processing_jobs
                    SET raw_file_name = ?,
                        processed_path = ?,
                        created_at = ?,
                        updated_at = ?,
                        status = ?,
                        started_at = ?,
                        completed_at = ?,
                        error = ?
                    WHERE raw_path = ?
                    """,
                    (
                        raw_file.name,
                        str(processed_file),
                        created_at,
                        now,
                        status,
                        started_at,
                        completed_at,
                        error,
                        raw_path_str,
                    ),
                )

            for row in rows:
                if row["raw_path"] in raw_files:
                    continue
                conn.execute("DELETE FROM processing_jobs WHERE raw_path = ?", (row["raw_path"],))

            conn.commit()

    def _set_terminal_state(self, raw_path: Path, status: str, error: Optional[str]) -> None:
        raw_file = raw_path.resolve()
        processed_file = processed_clip_path(raw_file, _settings.processed_videos_dir)
        now = _utc_now()
        completed_at = now if status == "completed" else None

        self._ensure_storage()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = ?,
                    processed_path = ?,
                    updated_at = ?,
                    completed_at = ?,
                    error = ?
                WHERE raw_path = ?
                """,
                (
                    status,
                    str(processed_file),
                    now,
                    completed_at,
                    error,
                    str(raw_file),
                ),
            )
            conn.commit()

    def _serialize_row(self, row: sqlite3.Row) -> dict:
        raw_path = Path(row["raw_path"])
        processed_path = Path(row["processed_path"])
        raw_exists = raw_path.exists()
        processed_exists = processed_path.exists() and row["status"] == "completed"

        return {
            "raw_file_name": row["raw_file_name"],
            "raw_path": str(raw_path),
            "raw_size_bytes": raw_path.stat().st_size if raw_exists else 0,
            "camera_id": row["camera_id"],
            "camera_name": row["camera_name"],
            "status": row["status"],
            "processed_file_name": processed_path.name if processed_exists else None,
            "processed_path": str(processed_path) if processed_exists else None,
            "processed_url": f"/processed/{processed_path.name}" if processed_exists else None,
            "processed_size_bytes": processed_path.stat().st_size if processed_exists else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
        }

    def _ensure_storage(self) -> None:
        _settings.recordings_dir.mkdir(parents=True, exist_ok=True)
        _settings.processed_videos_dir.mkdir(parents=True, exist_ok=True)
        _settings.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock, self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_STATUS_INDEX_SQL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(_settings.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn


processing_service = ProcessingService()
