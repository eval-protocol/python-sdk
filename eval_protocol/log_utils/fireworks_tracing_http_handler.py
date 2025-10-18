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
        self.gateway_base_url = (
            gateway_base_url or os.getenv("FW_TRACING_GATEWAY_BASE_URL") or "https://tracing.fireworks.ai"
        )
        self.rollout_id_env = rollout_id_env
        self._session = requests.Session()
        self._lock = threading.Lock()
        # Include Authorization header if FIREWORKS_API_KEY is available
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if api_key:
            try:
                self._session.headers.update({"Authorization": f"Bearer {api_key}"})
            except Exception:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self.gateway_base_url:
                return
            rollout_id = self._get_rollout_id(record)
            if not rollout_id:
                return
            payload = self._build_payload(record, rollout_id)
            base = self.gateway_base_url.rstrip("/")
            url = f"{base}/logs"
            # Optional debug prints to aid local diagnostics
            if os.environ.get("EP_DEBUG") == "true":
                try:
                    tags_val = payload.get("tags")
                    tags_len = len(tags_val) if isinstance(tags_val, list) else 0
                    msg_val = payload.get("message")
                    msg_preview = msg_val[:80] if isinstance(msg_val, str) else msg_val
                    print(f"[FW_LOG] POST {url} rollout_id={rollout_id} tags={tags_len} msg={msg_preview}")
                except Exception:
                    pass
            with self._lock:
                resp = self._session.post(url, json=payload, timeout=5)
            if os.environ.get("EP_DEBUG") == "true":
                try:
                    print(f"[FW_LOG] resp={resp.status_code}")
                except Exception:
                    pass
            # Fallback to /v1/logs if /logs is not found
            if resp is not None and getattr(resp, "status_code", None) == 404:
                alt = f"{base}/v1/logs"
                if os.environ.get("EP_DEBUG") == "true":
                    try:
                        tags_val = payload.get("tags")
                        tags_len = len(tags_val) if isinstance(tags_val, list) else 0
                        print(f"[FW_LOG] RETRY POST {alt} rollout_id={rollout_id} tags={tags_len}")
                    except Exception:
                        pass
                with self._lock:
                    resp2 = self._session.post(alt, json=payload, timeout=5)
                if os.environ.get("EP_DEBUG") == "true":
                    try:
                        print(f"[FW_LOG] retry resp={resp2.status_code}")
                    except Exception:
                        pass
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
