import asyncio
import time
from typing import Any, Dict, List, Optional, Callable

import requests

from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient
from eval_protocol.models import EvaluationRow, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import (
    DataLoaderConfig,
    ElasticsearchConfig,
)
from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
from .elasticsearch_setup import ElasticsearchSetup
from .tracing_utils import default_fireworks_output_data_loader, build_init_request, update_row_with_remote_trace
import logging

import os

logger = logging.getLogger(__name__)


def create_elasticsearch_config_from_env() -> ElasticsearchConfig:
    """Setup Elasticsearch config from environment variables."""
    url = os.getenv("ELASTICSEARCH_URL")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    index_name = os.getenv("ELASTICSEARCH_INDEX_NAME")

    if url is None:
        raise ValueError("ELASTICSEARCH_URL must be set")
    if api_key is None:
        raise ValueError("ELASTICSEARCH_API_KEY must be set")
    if index_name is None:
        raise ValueError("ELASTICSEARCH_INDEX_NAME must be set")
    return ElasticsearchConfig(
        url=url,
        api_key=api_key,
        index_name=index_name,
    )


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
        disable_elastic_search_setup: bool = False,
        elastic_search_config: Optional[ElasticsearchConfig] = None,
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
        self._disable_elastic_search_setup = disable_elastic_search_setup
        self._elastic_search_config = elastic_search_config

    def setup(self) -> None:
        if self._disable_elastic_search_setup:
            logger.info("Elasticsearch is disabled, skipping setup")
            return
        logger.info("Setting up Elasticsearch")
        self._elastic_search_config = self._setup_elastic_search()
        logger.info("Elasticsearch setup complete")

    def _setup_elastic_search(self) -> ElasticsearchConfig:
        """Set up Elasticsearch using the dedicated setup module."""
        setup = ElasticsearchSetup()
        return setup.setup_elasticsearch()

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

            init_payload = build_init_request(row, config, model_base_url, self._elastic_search_config)

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

            elasticsearch_client = (
                ElasticsearchClient(self._elastic_search_config) if self._elastic_search_config else None
            )

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
                        logger.info(
                            f"Server doesn't implement /status endpoint (404), stopping status polling for rollout {row.execution_metadata.rollout_id}"
                        )
                        continue_polling_status = False
                    else:
                        raise
                except Exception:
                    # For all other exceptions, raise them
                    raise

                if not elasticsearch_client:
                    continue

                search_results = elasticsearch_client.search_by_status_code_not_in(
                    row.execution_metadata.rollout_id, [Status.Code.RUNNING]
                )
                hits = search_results["hits"]["hits"] if search_results else []

                if hits:
                    # log all statuses found and update rollout status from the last hit
                    for hit in hits:
                        document = hit["_source"]
                        logger.info(
                            f"Found log for rollout {row.execution_metadata.rollout_id} with status code {document['status_code']}"
                        )
                        # Update rollout status from the document
                        if "status_code" in document:
                            row.rollout_status = Status(
                                code=Status.Code(document["status_code"]),
                                message=document.get("status_message", ""),
                                details=document.get("status_details", []),
                            )
                    logger.info("Stopping status polling for rollout %s", row.execution_metadata.rollout_id)
                    break

                await asyncio.sleep(poll_interval)
            else:
                logger.info(
                    f"Loop completed without breaking for {row.execution_metadata.rollout_id}, which means we timed out"
                )
                # Loop completed without breaking, which means we timed out
                row.rollout_status = Status.rollout_error(
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
