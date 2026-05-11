import yaml
from pathlib import Path
import logging

from app.utils.config_schema import AppConfig

logger = logging.getLogger("[ConfigLoader]")

def load_config(config_path: Path | str) -> AppConfig:
    """Read the app config YAML and validate schema via Pydantic."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
            
        validated_config = AppConfig(**raw_data)
        logger.info(f"Configuration successfully loaded and validated from {path}.")
        return validated_config
        
    except Exception as e:
        logger.error(f"Failed to load or validate configuration: {e}")
        raise
