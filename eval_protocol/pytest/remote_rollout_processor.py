import asyncio
import base64
import time
from typing import Any, Dict, List, Optional, Callable

import requests

from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient
from eval_protocol.models import EvaluationRow, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import (
    DataLoaderConfig,
    ElasticsearchConfig,
    InitRequest,
    RolloutMetadata,
)
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.quickstart.utils import filter_longest_conversation
from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
from .elasticsearch_setup import ElasticsearchSetup
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


def _build_fireworks_tracing_url(
    base_url: str, metadata: RolloutMetadata, completion_params_base_url: Optional[str] = None
) -> str:
    """Build a Fireworks tracing URL by appending rollout metadata to the base URL path,
    allowing the Fireworks tracing proxy to automatically tag traces.

    Format: {base_url}/rollout_id/{id}/invocation_id/{id}/experiment_id/{id}/run_id/{id}/row_id/{id}

    Args:
        base_url: Fireworks tracing proxy URL (we expect this to be https://tracing.fireworks.ai or
                  https://tracing.fireworks.ai/project_id/{project_id})
        metadata: Rollout metadata containing IDs to embed in the URL
        completion_params_base_url: Optional LLM base URL to encode and append to the final URL
    """
    url = (
        f"{base_url}/rollout_id/{metadata.rollout_id}"
        f"/invocation_id/{metadata.invocation_id}"
        f"/experiment_id/{metadata.experiment_id}"
        f"/run_id/{metadata.run_id}"
        f"/row_id/{metadata.row_id}"
    )

    if (
        completion_params_base_url
    ):  # The final URL is both tracing.fireworks.ai and the actual LLM base URL we want to use
        encoded_base_url = base64.urlsafe_b64encode(completion_params_base_url.encode()).decode()
        url = f"{url}/encoded_base_url/{encoded_base_url}"

    return url


def _default_output_data_loader(config: DataLoaderConfig) -> DynamicDataLoader:
    """Default output data loader that fetches traces from Fireworks tracing proxy.

    Args:
        config: Configuration containing rollout_id and optional model_base_url

    Returns:
        DynamicDataLoader configured to fetch and process traces
    """

    def fetch_traces() -> List[EvaluationRow]:
        base_url = config.model_base_url or "https://tracing.fireworks.ai"
        adapter = FireworksTracingAdapter(base_url=base_url)
        return adapter.get_evaluation_rows(tags=[f"rollout_id:{config.rollout_id}"], max_retries=5)

    return DynamicDataLoader(generators=[fetch_traces], preprocess_fn=filter_longest_conversation)


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
        self._output_data_loader = output_data_loader or _default_output_data_loader
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

            # Extract base_url from completion_params if provided. If we're using tracing.fireworks.ai, this base_url gets encoded and passed to LiteLLM inside the proxy.
            completion_params_base_url: Optional[str] = None
            if row.input_metadata and row.input_metadata.completion_params:
                completion_params_base_url = row.input_metadata.completion_params.get("base_url")
            if completion_params_base_url is None and config.completion_params:
                completion_params_base_url = config.completion_params.get("base_url")

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

            final_model_base_url = model_base_url
            if model_base_url and (
                model_base_url.startswith("https://tracing.fireworks.ai")
                or model_base_url.startswith("http://localhost")
            ):
                final_model_base_url = _build_fireworks_tracing_url(model_base_url, meta, completion_params_base_url)

            init_payload: InitRequest = InitRequest(
                model=model,
                messages=clean_messages,
                tools=row.tools,
                metadata=meta,
                model_base_url=final_model_base_url,
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

            # Update duration, regardless of termination
            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            if row.execution_metadata.rollout_id is None:
                raise ValueError("Rollout ID is required in RemoteRolloutProcessor")

            loader_config = DataLoaderConfig(
                rollout_id=row.execution_metadata.rollout_id, model_base_url=model_base_url
            )
            data_loader = self._output_data_loader(loader_config)

            def _load_data():
                return data_loader.load()

            results = await asyncio.to_thread(_load_data)

            output_rows: List[EvaluationRow] = [row for result in results for row in result.rows]

            if len(output_rows) == 0:  # Fallback to original row if no Remote data found
                row.rollout_status = Status(code=Status.Code.NOT_FOUND, message="No remote data found for rollout")
                return row
            elif len(output_rows) == 1:  # Return the remote row
                remote_row = output_rows[0]

                # if the remote_row has the same number of messages as the original row,
                # something went wrong
                if len(remote_row.messages) == len(row.messages):
                    row.rollout_status = Status.rollout_error(
                        "Rollout finished with the same number of messages as the original row"
                    )
                    return row

                row.messages = remote_row.messages
                row.tools = remote_row.tools
                row.input_metadata.session_data = remote_row.input_metadata.session_data
                row.execution_metadata = remote_row.execution_metadata
                return row
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
