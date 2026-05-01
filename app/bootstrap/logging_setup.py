import logging
import sys
from pathlib import Path
from app.utils.config_schema import LoggingConfig

def setup_logging(config: LoggingConfig):
    """Configure centralized logging for the application."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.to_file and config.file_path:
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-15s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True # Override any previous basicConfig
    )
    
    logging.getLogger("[Bootstrap]").info(f"Logging initialized at level: {config.level}")
