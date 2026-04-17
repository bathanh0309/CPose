# shared/io/job_store.py
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from shared.io.paths import get_run_dir


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def create_run(manifest: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:8]
    manifest["job_id"] = job_id  # Ensure job_id is in manifest
    run_dir = get_run_dir(job_id)

    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"job_id": job_id, "status": "queued", "progress": 0.0})

    return job_id


def update_status(job_id: str, status: str, progress: Optional[float] = None) -> None:
    run_dir = get_run_dir(job_id)
    status_path = run_dir / "status.json"
    current = _read_json(status_path) or {"job_id": job_id}
    current["status"] = status
    if progress is not None:
        current["progress"] = progress
    _write_json(status_path, current)


def get_status(job_id: str) -> Optional[Dict[str, Any]]:
    run_dir = get_run_dir(job_id)
    return _read_json(run_dir / "status.json")


def get_manifest(job_id: str) -> Optional[Dict[str, Any]]:
    run_dir = get_run_dir(job_id)
    return _read_json(run_dir / "manifest.json")


def get_results(job_id: str) -> Optional[Dict[str, Any]]:
    run_dir = get_run_dir(job_id)
    return _read_json(run_dir / "results.json")
