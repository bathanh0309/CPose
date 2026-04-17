# research/services/model_registry.py
from typing import Dict, Type

from research.services.experiment_engine import BaseExperimentEngine, DummyEngine

_REGISTRY: Dict[str, Type[BaseExperimentEngine]] = {
    "dummy": DummyEngine,
    # "ctrgcn": CTRGCNEngine,
    # "pose2id": Pose2IDEngine,
}

def get_engine_class(name: str) -> Type[BaseExperimentEngine]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown experiment engine: {name}")
