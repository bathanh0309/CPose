# shared/io/paths.py
import re
from pathlib import Path

_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.match(job_id):
        raise ValueError("Invalid job_id")
    return job_id


def get_project_root() -> Path:
    # shared/io/paths.py -> shared/io -> shared -> root
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_research_runs_dir() -> Path:
    d = get_data_dir() / "research_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_run_dir(job_id: str) -> Path:
    job_id = _validate_job_id(job_id)
    d = get_research_runs_dir() / f"run_{job_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
