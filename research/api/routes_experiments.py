from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel

router = APIRouter()

class ExperimentConfig(BaseModel):
    name: str
    model_name: str
    data_path: str
    params: Dict = {}

@router.post("/run")
async def run_experiment(config: ExperimentConfig):
    # Logic to trigger cpose.pipeline.multicam_analyzer or similar
    return {"status": "started", "experiment_name": config.name}

@router.get("/list")
async def list_experiments():
    return [{"id": "run_001", "status": "finished"}]
