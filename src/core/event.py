from pathlib import Path

from src.utils.io import append_jsonl, now_ms
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventBus:
    def __init__(self, output_jsonl=None, enabled=True):
        self.enabled = bool(enabled)
        self.output_jsonl = Path(output_jsonl) if output_jsonl else None
        if self.enabled and self.output_jsonl is not None:
            logger.info(f"Event log enabled: {self.output_jsonl}")
        else:
            logger.info("Event log disabled")

    def emit(self, event_type, payload):
        if not self.enabled or self.output_jsonl is None:
            return

        row = {
            "ts_ms": now_ms(),
            "type": event_type,
            "payload": payload
        }
        append_jsonl(self.output_jsonl, row)


class NullEventBus:
    def emit(self, event_type, payload):
        return
