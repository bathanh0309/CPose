# research/schemas/experiment.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Tên experiment, ví dụ: ctrgcn_vs_blockgcn")
    model_name: str = Field(..., description="Tên model research, map sang model_registry.yaml")
    input_source: str = Field(
        "raw_videos",
        description="raw_videos | output_labels | output_pose",
    )
    params: Dict[str, Any] = Field(default_factory=dict)


class ExperimentStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
