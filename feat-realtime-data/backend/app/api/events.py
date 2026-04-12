# API đọc danh sách sự kiện phát hiện và ghi hình từ SQLite.
from fastapi import APIRouter, Query

from app import database
from app.schemas import EventOut

router = APIRouter()


@router.get("/api/events", response_model=list[EventOut])
async def list_events(
    limit: int = Query(default=100, ge=1, le=1000)
) -> list[EventOut]:
    """Return recent events, newest first."""
    return await database.fetch_recent_events(limit=limit)
