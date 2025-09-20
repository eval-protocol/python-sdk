import asyncio
import time
from typing import Any, Dict, List, Optional

import requests

from eval_protocol.models import EvaluationRow
from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig


class RemoteRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that triggers a remote HTTP server to perform the rollout.

    Expected remote API:
    - POST {remote_base_url}/init
      Body: {
        "rollout_id": str,
        "model": str,
        "messages": list[dict],
        "tools": list[dict] | null,
        "metadata": {
          "invocation_id": str,
          "experiment_id": str,
          "rollout_id": str,
          "run_id": str | null,
          "row_id": str | null
        },
        "num_turns": int
      }
      Returns: {"ok": true}

    - GET {remote_base_url}/status?rollout_id=...
      Returns: {"terminated": bool, "info": {...}?}
    """

    def __init__(
        self,
        *,
        remote_base_url: Optional[str] = None,
        num_turns: int = 2,
        poll_interval: float = 1.0,
        timeout_seconds: float = 120.0,
    ):
        # Prefer constructor-provided configuration. These can be overridden via
        # config.kwargs at call time for backward compatibility.
        self._remote_base_url = remote_base_url
        self._num_turns = num_turns
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        tasks: List[asyncio.Task[EvaluationRow]] = []

        # Start with constructor values
        remote_base_url: Optional[str] = self._remote_base_url
        num_turns: int = self._num_turns
        poll_interval: float = self._poll_interval
        timeout_seconds: float = self._timeout_seconds

        # Backward compatibility: allow overrides via config.kwargs
        if config.kwargs:
            if remote_base_url is None:
                remote_base_url = config.kwargs.get("remote_base_url", remote_base_url)
            num_turns = int(config.kwargs.get("num_turns", num_turns))
            poll_interval = float(config.kwargs.get("poll_interval", poll_interval))
            timeout_seconds = float(config.kwargs.get("timeout_seconds", timeout_seconds))

        if not remote_base_url:
            raise ValueError("remote_base_url is required in RolloutProcessorConfig.kwargs for RemoteRolloutProcessor")

        async def _process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            # Build request metadata and payload
            meta: Dict[str, Any] = {
                "invocation_id": row.execution_metadata.invocation_id,
                "experiment_id": row.execution_metadata.experiment_id,
                "rollout_id": row.execution_metadata.rollout_id,
                "run_id": row.execution_metadata.run_id,
                "row_id": row.input_metadata.row_id,
            }

            model: Optional[str] = None
            if row.input_metadata and row.input_metadata.completion_params:
                model = row.input_metadata.completion_params.get("model")
            if model is None and config.completion_params:
                model = config.completion_params.get("model")
            if model is None:
                raise ValueError(
                    "Model must be provided in row.input_metadata.completion_params or config.completion_params"
                )

            # Strip non-OpenAI fields from messages before sending to remote
            allowed_message_fields = {"role", "content", "tool_calls", "tool_call_id", "name"}
            clean_messages = []
            for m in row.messages:
                md: Dict[str, Any]
                if hasattr(m, "model_dump"):
                    md = m.model_dump()  # type: ignore[assignment]
                elif isinstance(m, dict):
                    md = m  # type: ignore[assignment]
                else:
                    # Fallback to constructing a dict from Message-like object
                    md = {
                        "role": getattr(m, "role", None),
                        "content": getattr(m, "content", None),
                        "tool_calls": getattr(m, "tool_calls", None),
                        "tool_call_id": getattr(m, "tool_call_id", None),
                        "name": getattr(m, "name", None),
                    }
                clean_messages.append({k: v for k, v in md.items() if k in allowed_message_fields and v is not None})

            init_payload: Dict[str, Any] = {
                "rollout_id": row.execution_metadata.rollout_id,
                "model": model,
                "messages": clean_messages,
                "tools": row.tools,
                "metadata": meta,
                "num_turns": num_turns,
            }

            # Fire-and-poll
            def _post_init() -> None:
                url = f"{remote_base_url}/init"
                r = requests.post(url, json=init_payload, timeout=30)
                r.raise_for_status()

            await asyncio.to_thread(_post_init)

            terminated = False
            deadline = time.time() + timeout_seconds

            def _get_status() -> Dict[str, Any]:
                url = f"{remote_base_url}/status"
                r = requests.get(url, params={"rollout_id": row.execution_metadata.rollout_id}, timeout=15)
                r.raise_for_status()
                return r.json()

            while time.time() < deadline:
                try:
                    status = await asyncio.to_thread(_get_status)
                    terminated = bool(status.get("terminated", False))
                    if terminated:
                        break
                except Exception:
                    # transient errors; continue polling
                    pass
                await asyncio.sleep(poll_interval)

            # Update duration, regardless of termination
            row.execution_metadata.duration_seconds = time.perf_counter() - start_time
            return row

        for r in rows:
            tasks.append(asyncio.create_task(_process_row(r)))

        return tasks

    def cleanup(self) -> None:
        return None
