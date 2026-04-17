# research/services/experiment_engine.py
from abc import ABC, abstractmethod
from typing import Any, Dict

from shared.io import job_store

class BaseExperimentEngine(ABC):
    def __init__(self, manifest: Dict[str, Any], job_id: str):
        self.manifest = manifest
        self.job_id = job_id

    def update_status(self, status: str, progress: float | None = None) -> None:
        job_store.update_status(self.job_id, status, progress)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Thực thi experiment, trả về dict results (metrics, artifacts).
        """
        ...

class DummyEngine(BaseExperimentEngine):
    def run(self) -> Dict[str, Any]:
        self.update_status("running", 0.1)
        # TODO: gọi cpose, data loader,... ở đây
        self.update_status("running", 0.9)
        return {
            "metrics": {"accuracy": 0.0},
            "artifacts": [],
        }
