from typing import TypedDict, List, Optional

class ExperimentStatus(TypedDict):
    id: str
    status: str # "pending", "running", "done", "error"
    progress: float
    message: str

class ExperimentResult(TypedDict):
    id: str
    metrics: dict
    artifact_paths: List[str]
