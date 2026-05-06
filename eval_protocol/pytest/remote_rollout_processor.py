import asyncio
import time
from typing import List, Optional

import aiohttp

from eval_protocol.models import EvaluationRow, Status
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.exceptions import exception_for_status_code

from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
from .tracing_utils import default_fireworks_output_data_loader, build_init_request, update_row_with_remote_trace
from .utils import normalize_fireworks_model_for_litellm
import logging

import os

logger = logging.getLogger(__name__)


class RemoteRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that triggers a remote HTTP server to perform the rollout.

    Fetches traces from the Fireworks tracing proxy using rollout_id tags.

    See https://evalprotocol.io/tutorial/remote-rollout-processor for documentation.
    """

    def __init__(
        self,
        *,
        remote_base_url: Optional[str] = None,
        model_base_url: str = "https://tracing.fireworks.ai",
        poll_interval: float = 1.0,
        timeout_seconds: float = 120.0,
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
        self._tracing_adapter = FireworksTracingAdapter(base_url=self._model_base_url)
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_or_create_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)  # no total connection limit
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

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

            # Normalize Fireworks model names for LiteLLM routing
            config.completion_params = (
                normalize_fireworks_model_for_litellm(config.completion_params) or config.completion_params
            )
            row.input_metadata.completion_params = config.completion_params

            init_payload = build_init_request(row, config, model_base_url)

            # Fire-and-poll
            init_url = f"{remote_base_url}/init"

            timeout_init = aiohttp.ClientTimeout(total=600)

            try:
                session = self._get_or_create_session()
                async with session.post(init_url, json=init_payload.model_dump(), timeout=timeout_init) as resp:
                    if resp.status >= 500:
                        body = await resp.text()
                        raise ConnectionError(f"Remote /init returned server error (HTTP {resp.status}): {body}")
                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(f"Remote /init failed (HTTP {resp.status}): {body}")
                    resp.raise_for_status()
                    await resp.read()  # Drain the response body and release the connection back to the pool
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"The /init endpoint tried {init_url} with {init_payload.model_dump()} but timed out after 600 seconds."
                )

            deadline = time.time() + timeout_seconds

            while time.time() < deadline:
                session = self._get_or_create_session()
                status_result = await self._tracing_adapter.async_get_status(
                    session,
                    rollout_id=row.execution_metadata.rollout_id,
                )
                status = (status_result or {}).get("status")
                if isinstance(status, dict) and "code" in status:
                    status_code = status["code"]
                    if status_code == Status.Code.RUNNING:
                        await asyncio.sleep(poll_interval)
                        continue

                    logger.info(
                        "Found status for rollout %s with code %s",
                        row.execution_metadata.rollout_id,
                        status_code,
                    )

                    # /status only returns the code; backfill message/details/extras from Logs once.
                    status_message: str = ""
                    status_details: list = []
                    status_extras: dict = {}
                    completed_logs = await self._tracing_adapter.async_search_logs(
                        session, tags=[f"rollout_id:{row.execution_metadata.rollout_id}"]
                    )
                    # Pick the log row whose status code matches the terminal
                    # code from /status, so intermediate RUNNING checkpoints
                    # don't poison the backfill.
                    for log in completed_logs:
                        sd = log.get("status")
                        if isinstance(sd, dict) and sd.get("code") == status_code:
                            status_message = sd.get("message", "") or ""
                            status_details = sd.get("details", []) or []
                            raw_extras = log.get("extras") or {}
                            status_extras = {
                                k: v
                                for k, v in raw_extras.items()
                                if k not in ("logger_name", "level", "timestamp")
                            }
                            break

                    exception = exception_for_status_code(status_code, status_message)
                    if exception is not None:
                        raise exception

                    row.rollout_status = Status(
                        code=Status.Code(status_code),
                        message=status_message,
                        details=status_details,
                    )

                    if status_extras:
                        if row.execution_metadata.extra:
                            row.execution_metadata.extra.update(status_extras)
                        else:
                            row.execution_metadata.extra = status_extras

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

            row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time

            def _update_with_trace() -> None:
                return update_row_with_remote_trace(row, default_fireworks_output_data_loader, model_base_url)

            await asyncio.to_thread(_update_with_trace)  # Update row with remote trace in-place
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await _process_row(r)
                return result

        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks

    async def acleanup(self) -> None:
        """Async cleanup - preferred when you can await."""
        if self._session and not self._session.closed:
            await self._session.close()

    def cleanup(self) -> None:
        """Sync cleanup - best-effort, schedules close if event loop is running."""
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._session.close())
            except RuntimeError:
                logger.warning(
                    "RemoteRolloutProcessor.cleanup() called outside of async context. "
                    "Session may not be properly closed. Use `await processor.acleanup()` when possible."
                )
