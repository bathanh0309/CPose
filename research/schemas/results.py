# research/schemas/results.py
from typing import Dict, List
from pydantic import BaseModel


class ExperimentResultSummary(BaseModel):
    job_id: str
    metrics: Dict[str, float] = {}
    artifacts: List[str] = []
