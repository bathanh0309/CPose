from pathlib import Path

from src.utils.io import append_jsonl, now_ms
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventBus:
    def __init__(self, output_jsonl):
        self.output_jsonl = Path(output_jsonl)
        logger.info(f"Event log: {self.output_jsonl}")

    def emit(self, event_type, payload):
        row = {
            "ts_ms": now_ms(),
            "type": event_type,
            "payload": payload
        }
        append_jsonl(self.output_jsonl, row)
