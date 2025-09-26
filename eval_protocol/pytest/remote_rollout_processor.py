import asyncio
import time
from typing import Any, Dict, List, Optional, Callable

import requests

from eval_protocol.models import EvaluationRow, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import InitRequest, RolloutMetadata
from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig


def _attach_metadata_to_model_base_url(model_base_url: Optional[str], metadata: RolloutMetadata) -> Optional[str]:
    """
    Attach rollout metadata as query parameters to the model_base_url.

    Args:
        model_base_url: The base URL for the model API
        metadata: The rollout metadata containing IDs to attach

    Returns:
        The model_base_url with query parameters attached, or None if model_base_url is None
    """
    if model_base_url is None:
        return None

    # Parse existing query parameters
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed = urlparse(model_base_url)
    query_params = parse_qs(parsed.query)

    # Add rollout metadata as query parameters
    query_params.update(
        {
            "rollout_id": [metadata.rollout_id],
            "invocation_id": [metadata.invocation_id],
            "experiment_id": [metadata.experiment_id],
            "run_id": [metadata.run_id],
            "row_id": [metadata.row_id],
        }
    )

    # Rebuild the URL with new query parameters
    new_query = urlencode(query_params, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)


class RemoteRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that triggers a remote HTTP server to perform the rollout.

    The processor automatically attaches rollout metadata (rollout_id, invocation_id,
    experiment_id, run_id, row_id) as query parameters to the model_base_url when
    provided. This passes along rollout context to the remote server for use in
    LLM API calls.

    Example:
        If model_base_url is "https://api.openai.com/v1" and rollout_id is "abc123",
        the enhanced URL will be:
        "https://api.openai.com/v1?rollout_id=abc123&invocation_id=def456&..."

    See https://evalprotocol.io/tutorial/remote-rollout-processor for documentation.
    """

    def __init__(
        self,
        *,
        remote_base_url: Optional[str] = None,
        model_base_url: Optional[str] = None,
        poll_interval: float = 1.0,
        timeout_seconds: float = 120.0,
        output_data_loader: Callable[[str], DynamicDataLoader],
    ):
        """
        Initialize the remote rollout processor.

        Args:
            remote_base_url: Base URL of the remote rollout server (required)
            model_base_url: Base URL for LLM API calls. Will be enhanced with rollout
                metadata as query parameters to pass along rollout context to the remote server.
            poll_interval: Interval in seconds between status polls
            timeout_seconds: Maximum time to wait for rollout completion
            output_data_loader: Function to load rollout results by rollout_id
        """
        # Store configuration parameters
        self._remote_base_url = remote_base_url
        self._model_base_url = model_base_url
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        self._output_data_loader = output_data_loader

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        tasks: List[asyncio.Task[EvaluationRow]] = []

        if not self._remote_base_url:
            raise ValueError("remote_base_url is required for RemoteRolloutProcessor")

        async def _process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            if row.execution_metadata.invocation_id is None:
                raise ValueError("Invocation ID is required in RemoteRolloutProcessor")
            if row.execution_metadata.experiment_id is None:
                raise ValueError("Experiment ID is required in RemoteRolloutProcessor")
            if row.execution_metadata.rollout_id is None:
                raise ValueError("Rollout ID is required in RemoteRolloutProcessor")
            if row.execution_metadata.run_id is None:
                raise ValueError("Run ID is required in RemoteRolloutProcessor")
            if row.input_metadata.row_id is None:
                raise ValueError("Row ID is required in RemoteRolloutProcessor")

            # Build request metadata and payload
            meta: RolloutMetadata = RolloutMetadata(
                invocation_id=row.execution_metadata.invocation_id,
                experiment_id=row.execution_metadata.experiment_id,
                rollout_id=row.execution_metadata.rollout_id,
                run_id=row.execution_metadata.run_id,
                row_id=row.input_metadata.row_id,
            )

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

            if row.execution_metadata.rollout_id is None:
                raise ValueError("Rollout ID is required in RemoteRolloutProcessor")

            # Attach rollout metadata to model_base_url as query parameters
            # This passes along rollout context to the remote server for use in LLM calls
            enhanced_model_base_url = _attach_metadata_to_model_base_url(self._model_base_url, meta)

            init_payload: InitRequest = InitRequest(
                model=model,
                messages=clean_messages,
                tools=row.tools,
                metadata=meta,
                model_base_url=enhanced_model_base_url,
            )

            # Fire-and-poll
            def _post_init() -> None:
                url = f"{self._remote_base_url}/init"
                r = requests.post(url, json=init_payload.model_dump(), timeout=30)
                r.raise_for_status()

            await asyncio.to_thread(_post_init)

            terminated = False
            deadline = time.time() + self._timeout_seconds

            def _get_status() -> Dict[str, Any]:
                url = f"{self._remote_base_url}/status"
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
                await asyncio.sleep(self._poll_interval)

            # Update duration, regardless of termination
            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            if row.execution_metadata.rollout_id is None:
                raise ValueError("Rollout ID is required in RemoteRolloutProcessor")

            data_loader = self._output_data_loader(row.execution_metadata.rollout_id)

            def _load_data():
                return data_loader.load()

            results = await asyncio.to_thread(_load_data)

            output_rows: List[EvaluationRow] = [row for result in results for row in result.rows]

            if len(output_rows) == 0:  # Fallback to original row if no Langfuse data found
                row.rollout_status = Status(code=Status.Code.NOT_FOUND, message="No Langfuse data found for rollout")
                return row
            elif len(output_rows) == 1:  # Return the Langfuse row
                langfuse_row = output_rows[0]
                langfuse_row.input_metadata.completion_params = row.input_metadata.completion_params
                langfuse_row.eval_metadata = row.eval_metadata
                return langfuse_row
            else:
                raise ValueError("RemoteRolloutProcessor's output_data_loader should return exactly one row.")

        for r in rows:
            tasks.append(asyncio.create_task(_process_row(r)))

        return tasks

    def cleanup(self) -> None:
        return None
