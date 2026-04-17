# research/api/routes_experiments.py
from fastapi import APIRouter, HTTPException, BackgroundTasks

from research.schemas.experiment import ExperimentCreate, ExperimentStatus
from research.schemas.results import ExperimentResultSummary
from research.services import experiment_runner
from shared.io import job_store

router = APIRouter()


@router.post("/run", response_model=ExperimentStatus)
def run_experiment(req: ExperimentCreate, background_tasks: BackgroundTasks) -> ExperimentStatus:
    manifest = req.model_dump()
    job_id = experiment_runner.submit_experiment(manifest)

    # Đẩy job chạy nền, API trả về ngay lập tức
    background_tasks.add_task(experiment_runner.run_job, job_id)

    status = job_store.get_status(job_id)
    if not status:
        raise HTTPException(status_code=500, detail="Status file missing")
    return ExperimentStatus(**status)


@router.get("/status/{job_id}", response_model=ExperimentStatus)
def get_status(job_id: str) -> ExperimentStatus:
    status = job_store.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return ExperimentStatus(**status)


@router.get("/results/{job_id}", response_model=ExperimentResultSummary)
def get_results(job_id: str) -> ExperimentResultSummary:
    results = job_store.get_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    return ExperimentResultSummary(**results)
