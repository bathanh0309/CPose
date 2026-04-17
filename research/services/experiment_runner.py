# research/services/experiment_runner.py
from typing import Any, Dict

from shared.io import job_store
from shared.io.paths import get_run_dir
from research.services import model_registry
from research.services.experiment_engine import BaseExperimentEngine


def submit_experiment(manifest: Dict[str, Any]) -> str:
    """
    Tạo một run_xxx mới trong data/research_runs và lưu manifest/status.
    Trả về job_id.
    """
    job_id = job_store.create_run(manifest)
    return job_id


def build_engine_from_manifest(manifest: Dict[str, Any]) -> BaseExperimentEngine:
    engine_cls = model_registry.get_engine_class(manifest.get("model_name", "dummy"))
    job_id = manifest["job_id"]
    return engine_cls(manifest, job_id)


def run_job(job_id: str) -> None:
    """
    Hàm này chạy nền: đọc manifest, build engine, chạy, update status.
    """
    manifest = job_store.get_manifest(job_id)
    if not manifest:
        job_store.update_status(job_id, "error")
        return

    try:
        engine = build_engine_from_manifest(manifest)
        # Bắt đầu thực thi
        results = engine.run()
        
        # Lưu kết quả
        run_dir = get_run_dir(job_id)
        job_store._write_json(run_dir / "results.json", results)
        
        # Hoàn thành
        job_store.update_status(job_id, "completed", progress=1.0)
    except Exception as e:
        import logging
        logging.error(f"Experiment failed for job {job_id}: {str(e)}")
        job_store.update_status(job_id, f"error: {str(e)}")
