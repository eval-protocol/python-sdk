import asyncio
import time
from typing import Any, Dict, List, Optional, Callable

import requests

from eval_protocol.models import EvaluationRow, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import (
    DataLoaderConfig,
)
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.exceptions import exception_for_status_code

from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
from .tracing_utils import default_fireworks_output_data_loader, build_init_request, update_row_with_remote_trace
import logging

import os

logger = logging.getLogger(__name__)


class RemoteRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that triggers a remote HTTP server to perform the rollout.

    By default, fetches traces from the Fireworks tracing proxy using rollout_id tags.
    You can provide a custom output_data_loader for different tracing backends.

    See https://evalprotocol.io/tutorial/remote-rollout-processor for documentation.
    """

    def __init__(
        self,
        *,
        remote_base_url: Optional[str] = None,
        model_base_url: str = "https://tracing.fireworks.ai",
        poll_interval: float = 1.0,
        timeout_seconds: float = 120.0,
        output_data_loader: Optional[Callable[[DataLoaderConfig], DynamicDataLoader]] = None,
    ):
        # Prefer constructor-provided configuration. These can be overridden via
        # config.kwargs at call time for backward compatibility.
        self._remote_base_url = remote_base_url
        self._model_base_url = model_base_url
        if os.getenv("EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL"):
            self._remote_base_url = os.getenv("EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL")
        _ep_model_base_url = os.getenv("EP_MODEL_BASE_URL")
        if _ep_model_base_url:
            self._model_base_url = _ep_model_base_url
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        self._output_data_loader = output_data_loader or default_fireworks_output_data_loader
        self._tracing_adapter = FireworksTracingAdapter(base_url=self._model_base_url)

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        tasks: List[asyncio.Task[EvaluationRow]] = []

        # Start with constructor values
        remote_base_url: Optional[str] = self._remote_base_url
        model_base_url: str = self._model_base_url
        poll_interval: float = self._poll_interval
        timeout_seconds: float = self._timeout_seconds

        # Backward compatibility: allow overrides via config.kwargs
        if config.kwargs:
            if remote_base_url is None:
                remote_base_url = config.kwargs.get("remote_base_url", remote_base_url)
            poll_interval = float(config.kwargs.get("poll_interval", poll_interval))
            timeout_seconds = float(config.kwargs.get("timeout_seconds", timeout_seconds))

        if not remote_base_url:
            raise ValueError("remote_base_url is required in RolloutProcessorConfig.kwargs for RemoteRolloutProcessor")

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

            init_payload = build_init_request(row, config, model_base_url)

            # Fire-and-poll
            def _post_init() -> None:
                url = f"{remote_base_url}/init"
                try:
                    r = requests.post(url, json=init_payload.model_dump(), timeout=300)
                    r.raise_for_status()
                except requests.exceptions.Timeout:
                    raise TimeoutError(
                        f"The /init endpoint tried {url} with {init_payload.model_dump()} but timed out after 300 seconds."
                    )

            await asyncio.to_thread(_post_init)

            terminated = False
            deadline = time.time() + timeout_seconds

            def _get_status() -> Dict[str, Any]:
                url = f"{remote_base_url}/status"
                r = requests.get(url, params={"rollout_id": row.execution_metadata.rollout_id}, timeout=15)
                r.raise_for_status()
                return r.json()

            continue_polling_status = True
            while time.time() < deadline:
                try:
                    if continue_polling_status:
                        status = await asyncio.to_thread(_get_status)
                        terminated = bool(status.get("terminated", False))
                        if terminated:
                            break
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        # 404 means server doesn't implement /status endpoint, stop polling
                        logger.debug(
                            f"Server doesn't implement /status endpoint (404), stopping status polling for rollout {row.execution_metadata.rollout_id}"
                        )
                        continue_polling_status = False
                    else:
                        raise
                except Exception:
                    # For all other exceptions, raise them
                    raise

                # Search Fireworks tracing logs for completion (run in thread to avoid blocking event loop)
                completed_logs = await asyncio.to_thread(
                    self._tracing_adapter.search_logs, tags=[f"rollout_id:{row.execution_metadata.rollout_id}"]
                )
                # Filter for logs that actually have status information
                status_logs = []
                for log in completed_logs:
                    status_dict = log.get("status")
                    if status_dict and isinstance(status_dict, dict) and "code" in status_dict:
                        status_logs.append(log)

                if status_logs:
                    # Use the first log with status information
                    status_log = status_logs[0]
                    status_dict = status_log.get("status")

                    logger.info(
                        f"Found status log for rollout {row.execution_metadata.rollout_id}: {status_log.get('message', '')}"
                    )

                    status_code = status_dict.get("code")
                    status_message = status_dict.get("message", "")
                    status_details = status_dict.get("details", [])

                    logger.info(
                        f"Found Fireworks log for rollout {row.execution_metadata.rollout_id} with status code {status_code}"
                    )

                    # Create and raise exception if appropriate, preserving original message
                    exception = exception_for_status_code(status_code, status_message)
                    if exception is not None:
                        raise exception

                    row.rollout_status = Status(
                        code=Status.Code(status_code),
                        message=status_message,
                        details=status_details,
                    )

                    logger.info("Stopping polling for rollout %s", row.execution_metadata.rollout_id)
                    break

                await asyncio.sleep(poll_interval)
            else:
                logger.info(
                    f"Loop completed without breaking for {row.execution_metadata.rollout_id}, which means we timed out"
                )
                # Loop completed without breaking, which means we timed out
                row.rollout_status = Status.rollout_deadline_exceeded_error(
                    f"Rollout {row.execution_metadata.rollout_id} timed out after {timeout_seconds} seconds"
                )

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            def _update_with_trace() -> None:
                return update_row_with_remote_trace(row, self._output_data_loader, model_base_url)

            await asyncio.to_thread(_update_with_trace)  # Update row with remote trace in-place
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await _process_row(r)
                return result

        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks

    def cleanup(self) -> None:
        return None
