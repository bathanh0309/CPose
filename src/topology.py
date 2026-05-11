from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.common.errors import ErrorCode


@dataclass(slots=True)
class Zone:
    name: str
    polygon: list[list[float]]


@dataclass(slots=True)
class CameraNode:
    camera_id: str
    role: str | None
    location: str | None
    entry_zones: list[Zone]
    exit_zones: list[Zone]


@dataclass(slots=True)
class CameraTransition:
    from_camera: str
    to_camera: str
    min_sec: float
    max_sec: float
    from_zone: str | None
    to_zone: str | None
    confidence: float = 1.0


@dataclass(slots=True)
class CameraTopology:
    cameras: dict[str, CameraNode]
    transitions: list[CameraTransition]


def _zones(rows: Any) -> list[Zone]:
    zones: list[Zone] = []
    for row in rows or []:
        if isinstance(row, str):
            zones.append(Zone(row, []))
        elif isinstance(row, dict):
            zones.append(Zone(str(row.get("name", "")), row.get("polygon", []) or []))
    return [zone for zone in zones if zone.name]


def load_camera_topology(path: str | Path | None) -> CameraTopology:
    if path is None or not Path(path).exists():
        print(f"[WARN] Camera topology not found: {path}")
        return CameraTopology(cameras={}, transitions=[])
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:
        print(f"[WARN] Could not load topology {path}: {exc}")
        return CameraTopology(cameras={}, transitions=[])

    cameras: dict[str, CameraNode] = {}
    for camera_id, row in (payload.get("cameras") or {}).items():
        row = row or {}
        cameras[str(camera_id)] = CameraNode(
            camera_id=str(camera_id),
            role=row.get("role"),
            location=row.get("location"),
            entry_zones=_zones(row.get("entry_zones")),
            exit_zones=_zones(row.get("exit_zones")),
        )
    transitions: list[CameraTransition] = []
    for row in payload.get("transitions") or []:
        transitions.append(CameraTransition(
            from_camera=str(row.get("from") or row.get("from_camera")),
            to_camera=str(row.get("to") or row.get("to_camera")),
            min_sec=float(row.get("min_sec", 0.0)),
            max_sec=float(row.get("max_sec", 10**9)),
            from_zone=row.get("from_zone"),
            to_zone=row.get("to_zone"),
            confidence=float(row.get("confidence", 1.0)),
        ))
    return CameraTopology(cameras=cameras, transitions=transitions)


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = float(current[0]), float(current[1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def bbox_bottom_center(bbox: list[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def infer_zone(camera_id: str, bbox: list[float], zone_type: Literal["entry", "exit"], topology: CameraTopology) -> str | None:
    node = topology.cameras.get(camera_id)
    if not node:
        return None
    zones = node.entry_zones if zone_type == "entry" else node.exit_zones
    point = bbox_bottom_center(bbox)
    for zone in zones:
        if zone.polygon and point_in_polygon(point, zone.polygon):
            return zone.name
    return zones[0].name if len(zones) == 1 and not zones[0].polygon else None


def get_valid_transitions(topology: CameraTopology, from_camera: str, to_camera: str) -> list[CameraTransition]:
    return [t for t in topology.transitions if t.from_camera == from_camera and t.to_camera == to_camera]


def is_transition_allowed(topology: CameraTopology, from_camera: str, to_camera: str, delta_time_sec: float) -> bool:
    return bool(topology_score(topology, from_camera, to_camera, delta_time_sec).get("allowed"))


def topology_score(
    topology: CameraTopology,
    from_camera: str,
    to_camera: str,
    delta_time_sec: float | None = None,
    exit_zone: str | None = None,
    entry_zone: str | None = None,
    *,
    delta_sec: float | None = None,
) -> dict[str, Any]:
    if delta_time_sec is None:
        delta_time_sec = 0.0 if delta_sec is None else delta_sec
    if from_camera == to_camera:
        return {
            "allowed": True,
            "score": 1.0,
            "transition": f"{from_camera}->{to_camera}",
            "delta_time_sec": float(delta_time_sec),
            "failure_reason": ErrorCode.OK.value,
        }
    transitions = get_valid_transitions(topology, from_camera, to_camera)
    if not transitions:
        return {
            "allowed": False,
            "score": 0.0,
            "transition": None,
            "delta_time_sec": float(delta_time_sec),
            "failure_reason": ErrorCode.TOPOLOGY_CONFLICT.value,
        }
    time_candidates = [t for t in transitions if t.min_sec <= delta_time_sec <= t.max_sec]
    if not time_candidates:
        return {
            "allowed": False,
            "score": 0.0,
            "transition": f"{from_camera}->{to_camera}",
            "delta_time_sec": float(delta_time_sec),
            "failure_reason": ErrorCode.TIME_WINDOW_CONFLICT.value,
        }
    best = time_candidates[0]
    score = best.confidence
    if best.from_zone and exit_zone:
        score *= 1.0 if best.from_zone == exit_zone else 0.3
    elif best.from_zone and not exit_zone:
        score *= 0.8
    if best.to_zone and entry_zone:
        score *= 1.0 if best.to_zone == entry_zone else 0.3
    elif best.to_zone and not entry_zone:
        score *= 0.8
    allowed = score > 0.31
    return {
        "allowed": allowed,
        "score": round(float(score), 4),
        "transition": f"{best.from_camera}->{best.to_camera}",
        "delta_time_sec": float(delta_time_sec),
        "failure_reason": ErrorCode.OK.value if allowed else ErrorCode.TOPOLOGY_CONFLICT.value,
    }
