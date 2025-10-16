import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, cast

import requests


class FireworksTracingHttpHandler(logging.Handler):
    """Logging handler that posts structured logs to tracing.fireworks gateway /logs endpoint."""

    def __init__(self, gateway_base_url: Optional[str] = None, rollout_id_env: str = "EP_ROLLOUT_ID") -> None:
        super().__init__()
        self.gateway_base_url = gateway_base_url or os.getenv("FW_TRACING_GATEWAY_BASE_URL")
        self.rollout_id_env = rollout_id_env
        self._session = requests.Session()
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self.gateway_base_url:
                return
            rollout_id = self._get_rollout_id(record)
            if not rollout_id:
                return
            payload = self._build_payload(record, rollout_id)
            url = f"{self.gateway_base_url.rstrip('/')}/logs"
            with self._lock:
                self._session.post(url, json=payload, timeout=5)
        except Exception:
            # Avoid raising exceptions from logging
            self.handleError(record)

    def _get_rollout_id(self, record: logging.LogRecord) -> Optional[str]:
        if hasattr(record, "rollout_id") and cast(Any, getattr(record, "rollout_id")) is not None:
            return str(cast(Any, getattr(record, "rollout_id")))
        return os.getenv(self.rollout_id_env)

    def _build_payload(self, record: logging.LogRecord, rollout_id: str) -> Dict[str, Any]:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        message = record.getMessage()
        tags: List[str] = [f"rollout_id:{rollout_id}"]
        # Optional additional tags
        if hasattr(record, "experiment_id") and cast(Any, getattr(record, "experiment_id")):
            tags.append(f"experiment_id:{cast(Any, getattr(record, 'experiment_id'))}")
        if hasattr(record, "run_id") and cast(Any, getattr(record, "run_id")):
            tags.append(f"run_id:{cast(Any, getattr(record, 'run_id'))}")
        # Groupwise list of rollout_ids
        if hasattr(record, "rollout_ids") and cast(Any, getattr(record, "rollout_ids")):
            try:
                for rid in cast(List[str], getattr(record, "rollout_ids")):
                    tags.append(f"rollout_id:{rid}")
            except Exception:
                pass
        program = cast(Optional[str], getattr(record, "program", None)) or "eval_protocol"
        status_val = cast(Any, getattr(record, "status", None))
        status = status_val if isinstance(status_val, str) else None
        # Capture optional structured status fields if present
        metadata: Dict[str, Any] = {}
        status_code = cast(Any, getattr(record, "status_code", None))
        if isinstance(status_code, int):
            metadata["status_code"] = status_code
        status_message = cast(Any, getattr(record, "status_message", None))
        if isinstance(status_message, str):
            metadata["status_message"] = status_message
        status_details = getattr(record, "status_details", None)
        if status_details is not None:
            metadata["status_details"] = status_details
        extra_metadata = cast(Any, getattr(record, "metadata", None))
        if isinstance(extra_metadata, dict):
            metadata.update(extra_metadata)
        return {
            "program": program,
            "status": status,
            "message": message,
            "tags": tags,
            "metadata": metadata or None,
            "extras": {
                "logger_name": record.name,
                "level": record.levelname,
                "timestamp": timestamp,
            },
        }
