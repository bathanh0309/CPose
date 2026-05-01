"""
Module 5 — Cross-Camera Re-Identification (ReID)
=================================================
Thuật toán TFCS-PAR (Time-First Cross-Camera Sequential Pose-ADL-ReID)

Quy tắc gán Global ID:
  - Đọc tất cả video trong data/multicam, parse timestamp từ tên file.
  - Sort theo thứ tự: (year, month, day, hour, minute, second, cam_index).
  - Người đầu tiên xuất hiện ở cam1 hoặc cam2 → GID-001, GID-002, ...
  - Khi người xuất hiện lại ở camera khác (theo topology + thời gian),
    ưu tiên giữ nguyên GID cũ (không tạo ID mới).
  - Global ID xuyên suốt tất cả camera — đây là module DUY NHẤT được
    phép vẽ GID-XXX lên overlay video.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.geometry import bbox_iou  # noqa: F401 (available for callers)
from src.common.paths import resolve_path


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"cam(\d+)[_\-](\d{4})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})",
    re.IGNORECASE,
)


@dataclass
class ClipInfo:
    path: Path
    cam_index: int
    clip_dt: datetime

    @property
    def cam_id(self) -> str:
        return f"cam{self.cam_index}"

    @property
    def sort_key(self) -> tuple:
        dt = self.clip_dt
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, self.cam_index)


def parse_clip(path: Path) -> ClipInfo | None:
    """Parse cam_index and datetime from filename like cam1_2026-01-29_16-26-25.mp4."""
    m = _TIMESTAMP_RE.search(path.stem)
    if not m:
        return None
    cam_index = int(m.group(1))
    year, month, day = int(m.group(2)), int(m.group(3)), int(m.group(4))
    hour, minute, second = int(m.group(5)), int(m.group(6)), int(m.group(7))
    try:
        clip_dt = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    return ClipInfo(path=path, cam_index=cam_index, clip_dt=clip_dt)


def list_sorted_clips(multicam_dir: str | Path) -> list[ClipInfo]:
    """Return all mp4 clips sorted by (timestamp, cam_index) per .algorithm.md §3.2."""
    directory = resolve_path(multicam_dir)
    clips: list[ClipInfo] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() == ".mp4":
            info = parse_clip(p)
            if info:
                clips.append(info)
            else:
                print(f"  [WARN] Cannot parse timestamp from '{p.name}', skipping.")
    clips.sort(key=lambda c: c.sort_key)
    return clips


# ---------------------------------------------------------------------------
# Camera topology (per .algorithm.md §4)
# ---------------------------------------------------------------------------

# Adjacent cameras that a person can reasonably walk between
# cam_index -> list of (next_cam_index, min_sec, max_sec)
TOPOLOGY: dict[int, list[tuple[int, int, int]]] = {
    1: [(2, 0, 60)],
    2: [(1, 0, 60), (3, 0, 60)],
    3: [(2, 10, 120), (4, 20, 180)],
    4: [(3, 20, 180), (4, 5, 300)],   # cam4->cam4 = room re-entry
}


def plausible_transition(from_cam: int, to_cam: int, elapsed_sec: float) -> bool:
    """Return True if transition from→to within elapsed_sec is spatially valid."""
    for next_cam, min_s, max_s in TOPOLOGY.get(from_cam, []):
        if next_cam == to_cam and min_s <= elapsed_sec <= max_s:
            return True
    return False


# ---------------------------------------------------------------------------
# Appearance feature: simple body-crop HSV histogram
# ---------------------------------------------------------------------------

def _body_feature(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _feat_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5  # neutral when no feature available
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


def _height_ratio(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    return h / w if w > 0 else 0.0


# ---------------------------------------------------------------------------
# Global Person Table
# ---------------------------------------------------------------------------

@dataclass
class GlobalPerson:
    gid: int
    status: str = "ACTIVE"          # ACTIVE | PENDING_TRANSFER | IN_ROOM | DORMANT | CLOSED
    last_cam: int = 0
    last_time: datetime | None = None
    last_bbox: list[float] = field(default_factory=list)
    last_adl: str = "unknown"
    last_body_feat: Any = None      # np.ndarray or None
    last_height_ratio: float = 0.0
    history: list[dict] = field(default_factory=list)


class GlobalPersonTable:
    """Maintains cross-camera Global IDs (GIDs). Minimizes new ID creation."""

    def __init__(self) -> None:
        self._table: dict[int, GlobalPerson] = {}
        self._next_gid: int = 1

    def _new_gid(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def all_active_candidates(self, cam_index: int, current_time: datetime) -> list[GlobalPerson]:
        """Return pending/dormant persons that could plausibly appear in cam_index now."""
        candidates: list[GlobalPerson] = []
        for person in self._table.values():
            if person.status == "CLOSED":
                continue
            if person.last_time is None:
                continue
            elapsed = (current_time - person.last_time).total_seconds()
            if elapsed < 0:
                continue
            if person.status == "ACTIVE" and person.last_cam == cam_index:
                candidates.append(person)
            elif plausible_transition(person.last_cam, cam_index, elapsed):
                candidates.append(person)
        return candidates

    def match_or_create(
        self,
        bbox: list[float],
        frame: np.ndarray,
        cam_index: int,
        current_time: datetime,
        adl_label: str = "unknown",
    ) -> tuple[GlobalPerson, str]:
        """Match detection to existing GID or create new one. Returns (person, match_status)."""
        body_feat = _body_feature(frame, bbox)
        h_ratio = _height_ratio(bbox)

        candidates = self.all_active_candidates(cam_index, current_time)

        best_person: GlobalPerson | None = None
        best_score = 0.0
        for cand in candidates:
            elapsed = (current_time - cand.last_time).total_seconds() if cand.last_time else 9999
            # S_body (appearance)
            s_body = _feat_sim(body_feat, cand.last_body_feat)
            # S_height
            s_height = max(0.0, 1.0 - abs(h_ratio - cand.last_height_ratio) * 2.0)
            # S_time_topology
            if cand.last_cam == cam_index:
                s_topo = 1.0
            elif plausible_transition(cand.last_cam, cam_index, elapsed):
                s_topo = max(0.0, 1.0 - elapsed / 300.0)
            else:
                s_topo = 0.0
            # ADL continuity bonus
            s_adl = 0.1 if cand.last_adl == adl_label else 0.0
            # Scoring (no face available, so re-weight per §13.2)
            score = 0.35 * s_body + 0.20 * s_height + 0.35 * s_topo + 0.10 * s_adl
            if score > best_score:
                best_score = score
                best_person = cand

        # Single pending candidate always wins (§10.3 policy)
        if len(candidates) == 1 and best_score >= 0.30:
            match_status = "ACTIVE" if best_score >= 0.75 else "SOFT_MATCH"
        elif best_score >= 0.75:
            match_status = "ACTIVE"
        elif best_score >= 0.60:
            match_status = "SOFT_MATCH"
        else:
            best_person = None
            match_status = "ACTIVE"

        if best_person is None:
            gid = self._new_gid()
            person = GlobalPerson(gid=gid)
            self._table[gid] = person
        else:
            person = best_person

        # Update state
        person.status = match_status if match_status == "SOFT_MATCH" else "ACTIVE"
        person.last_cam = cam_index
        person.last_time = current_time
        person.last_bbox = bbox
        person.last_adl = adl_label
        person.last_body_feat = body_feat
        person.last_height_ratio = h_ratio
        person.history.append({
            "cam": cam_index,
            "time": current_time.isoformat(),
            "adl": adl_label,
            "match_status": match_status,
        })
        return person, match_status

    def mark_dormant_missing(self, active_gids: set[int], current_time: datetime) -> None:
        """Mark persons not seen in this clip as DORMANT if within TTL."""
        for person in self._table.values():
            if person.gid in active_gids or person.status == "CLOSED":
                continue
            if person.status == "ACTIVE":
                person.status = "PENDING_TRANSFER"
            elif person.status == "PENDING_TRANSFER":
                if person.last_time:
                    elapsed = (current_time - person.last_time).total_seconds()
                    if elapsed > 300:
                        person.status = "DORMANT"

    def to_dict(self) -> dict:
        result = {}
        for gid, p in self._table.items():
            result[f"GID-{gid:03d}"] = {
                "status": p.status,
                "last_cam": p.last_cam,
                "last_time": p.last_time.isoformat() if p.last_time else None,
                "last_adl": p.last_adl,
                "history": p.history,
            }
        return result
