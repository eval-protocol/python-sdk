import asyncio
import subprocess
import time
from typing import Any, Dict, List, Optional, Callable

from dotenv import load_dotenv
import requests

from eval_protocol.directory_utils import find_eval_protocol_dir
from eval_protocol.models import EvaluationRow, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import ElasticSearchConfig, InitRequest, RolloutMetadata
from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
import logging

import os

logger = logging.getLogger(__name__)


class RemoteRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that triggers a remote HTTP server to perform the rollout.

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
        disable_elastic_search: bool = False,
        elastic_search_config: Optional[ElasticSearchConfig] = None,
    ):
        # Prefer constructor-provided configuration. These can be overridden via
        # config.kwargs at call time for backward compatibility.
        self._remote_base_url = remote_base_url
        self._model_base_url = model_base_url
        if os.getenv("EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL"):
            self._remote_base_url = os.getenv("EP_REMOTE_ROLLOUT_PROCESSOR_BASE_URL")
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        self._output_data_loader = output_data_loader
        self._disable_elastic_search = disable_elastic_search
        self._elastic_search_config = elastic_search_config

    def setup(self) -> None:
        if self._disable_elastic_search:
            logger.info("Elasticsearch is disabled, skipping setup")
            return
        logger.info("Setting up Elasticsearch")
        self._elastic_search_config = self._setup_elastic_search()
        logger.info("Elasticsearch setup complete")

    def _parse_elastic_env_file(self, env_file_path: str) -> ElasticSearchConfig:
        """Parse ES_LOCAL_API_KEY and ES_LOCAL_URL from .env file."""
        loaded = load_dotenv(env_file_path)
        if not loaded:
            raise RuntimeError("Failed to load .env file")
        api_key = os.getenv("ES_LOCAL_API_KEY")
        url = os.getenv("ES_LOCAL_URL")
        if not url or not api_key:
            raise RuntimeError("Failed to parse ES_LOCAL_API_KEY and ES_LOCAL_URL from .env file")
        return ElasticSearchConfig(url=url, api_key=api_key)

    def _setup_elastic_search(self) -> ElasticSearchConfig:
        eval_protocol_dir = find_eval_protocol_dir()
        elastic_start_local_dir = os.path.join(eval_protocol_dir, "elastic-start-local")
        env_file_path = os.path.join(elastic_start_local_dir, ".env")

        # if elastic-start-local directory exists, return the config
        if os.path.exists(elastic_start_local_dir):
            # run start.sh in the elastic-start-local directory
            from eval_protocol.utils.subprocess_utils import run_script_and_wait

            run_script_and_wait(
                script_name="start.sh",
                working_directory=elastic_start_local_dir,
                inherit_stdout=True,
            )
            return self._parse_elastic_env_file(env_file_path)

        # run Elasticsearch start-local script: "curl -fsSL https://elastic.co/start-local | sh -s -- --esonly"
        process = subprocess.Popen(
            ["sh", "-c", "curl -fsSL https://elastic.co/start-local | sh -s -- --esonly"],
            cwd=eval_protocol_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError("Failed to start Elasticsearch")

        return self._parse_elastic_env_file(env_file_path)

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        tasks: List[asyncio.Task[EvaluationRow]] = []

        # Start with constructor values
        remote_base_url: Optional[str] = self._remote_base_url
        model_base_url: Optional[str] = self._model_base_url
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

            init_payload: InitRequest = InitRequest(
                model=model,
                messages=clean_messages,
                tools=row.tools,
                metadata=meta,
                model_base_url=model_base_url,
                elastic_search_config=self._elastic_search_config,
            )

            # Fire-and-poll
            def _post_init() -> None:
                url = f"{remote_base_url}/init"
                try:
                    r = requests.post(url, json=init_payload.model_dump(), timeout=30)
                    r.raise_for_status()
                except requests.exceptions.Timeout:
                    raise TimeoutError(
                        "The /init endpoint timed out after 30 seconds. "
                        "CRITICAL: The /init endpoint must return immediately (within 30s) and NOT block on rollout execution. "
                        "Your remote server should:\n"
                        "1. Accept the /init request and return a 200 response immediately\n"
                        "2. Process the actual rollout asynchronously in the background\n"
                        "3. Use the /status endpoint to report progress\n"
                        "For Python/Node.js: Start a separate process per rollout to avoid blocking the /init response."
                    )

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
            else:
                # Loop completed without breaking, which means we timed out
                row.rollout_status = Status.rollout_error(
                    f"Rollout {row.execution_metadata.rollout_id} timed out after {timeout_seconds} seconds"
                )

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
                # merge dataset_info dicts on input_metadata
                if langfuse_row.input_metadata.dataset_info and row.input_metadata.dataset_info:
                    langfuse_row.input_metadata.dataset_info = {
                        **row.input_metadata.dataset_info,
                        **langfuse_row.input_metadata.dataset_info,
                    }
                elif row.input_metadata.dataset_info:
                    langfuse_row.input_metadata.dataset_info = row.input_metadata.dataset_info
                langfuse_row.eval_metadata = row.eval_metadata
                langfuse_row.ground_truth = row.ground_truth
                return langfuse_row
            else:
                raise ValueError("RemoteRolloutProcessor's output_data_loader should return exactly one row.")

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await _process_row(r)
                return result

        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks

    def cleanup(self) -> None:
        return None
