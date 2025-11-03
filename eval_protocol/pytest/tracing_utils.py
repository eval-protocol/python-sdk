"""
Shared utilities for rollout processors.
"""

import base64
import os
from typing import Any, Callable, Dict, List, Optional

from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Status
from eval_protocol.utils.evaluation_row_utils import filter_longest_conversation
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig, RolloutMetadata, InitRequest
from eval_protocol.pytest.types import RolloutProcessorConfig


def default_fireworks_output_data_loader(config: DataLoaderConfig) -> DynamicDataLoader:
    """Default output data loader that fetches traces from Fireworks tracing proxy."""

    def fetch_traces() -> List[EvaluationRow]:
        base_url = config.model_base_url or "https://tracing.fireworks.ai"
        adapter = FireworksTracingAdapter(base_url=base_url)
        return adapter.get_evaluation_rows(tags=[f"rollout_id:{config.rollout_id}"], max_retries=5)

    return DynamicDataLoader(generators=[fetch_traces], preprocess_fn=filter_longest_conversation)


def build_fireworks_tracing_url(
    base_url: str, metadata: RolloutMetadata, completion_params_base_url: Optional[str] = None
) -> str:
    """Build a Fireworks tracing URL by appending rollout metadata to the base URL path,
    allowing the Fireworks tracing proxy to automatically tag traces.

    Format: {base_url}/rollout_id/{id}/invocation_id/{id}/experiment_id/{id}/run_id/{id}/row_id/{id}

    Args:
        base_url: Fireworks tracing proxy URL (e.g., https://tracing.fireworks.ai)
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

    if completion_params_base_url:
        encoded_base_url = base64.urlsafe_b64encode(completion_params_base_url.encode()).decode()
        url = f"{url}/encoded_base_url/{encoded_base_url}"

    return url


def build_init_request(
    row: EvaluationRow,
    config: RolloutProcessorConfig,
    model_base_url: str,
) -> InitRequest:
    """Build an InitRequest from an EvaluationRow and config (shared logic)."""
    # Validation
    if row.execution_metadata.invocation_id is None:
        raise ValueError("Invocation ID is required")
    if row.execution_metadata.experiment_id is None:
        raise ValueError("Experiment ID is required")
    if row.execution_metadata.rollout_id is None:
        raise ValueError("Rollout ID is required")
    if row.execution_metadata.run_id is None:
        raise ValueError("Run ID is required")
    if row.input_metadata.row_id is None:
        raise ValueError("Row ID is required")

    # Build metadata
    meta = RolloutMetadata(
        invocation_id=row.execution_metadata.invocation_id,
        experiment_id=row.execution_metadata.experiment_id,
        rollout_id=row.execution_metadata.rollout_id,
        run_id=row.execution_metadata.run_id,
        row_id=row.input_metadata.row_id,
    )

    # Build completion_params from row and config
    completion_params_dict: Dict[str, Any] = {}

    # Start with config-level completion_params
    if config.completion_params and isinstance(config.completion_params, dict):
        completion_params_dict.update(config.completion_params)

    # Override with row-specific completion_params
    if row.input_metadata and row.input_metadata.completion_params:
        row_cp = row.input_metadata.completion_params
        if isinstance(row_cp, dict):
            completion_params_dict.update(row_cp)

    # Validate model is present
    if not completion_params_dict.get("model"):
        raise ValueError("Model must be provided in completion_params")

    # Extract base_url from completion_params
    completion_params_base_url: Optional[str] = completion_params_dict.get("base_url")

    # Strip non-OpenAI fields from messages
    # Use dump_mdoel_for_chat_completion_request() to automatically exclude unsupported fields (weight, control_plane_step, reasoning_content)
    clean_messages = []
    for m in row.messages:
        md: Dict[str, Any]
        if hasattr(m, "dump_mdoel_for_chat_completion_request"):
            # Use the Message method that automatically filters unsupported fields
            md = m.dump_mdoel_for_chat_completion_request()
        elif hasattr(m, "model_dump"):
            md = m.model_dump()
        elif isinstance(m, dict):
            md = m
        else:
            # Fallback to constructing a dict from Message-like object
            md = {
                "role": getattr(m, "role", None),
                "content": getattr(m, "content", None),
                "tool_calls": getattr(m, "tool_calls", None),
                "tool_call_id": getattr(m, "tool_call_id", None),
                "name": getattr(m, "name", None),
            }
        # Additional filtering to ensure only allowed fields are kept (already handled by dump_mdoel_for_chat_completion_request for Message objects)
        allowed_message_fields = {"role", "content", "tool_calls", "tool_call_id", "name"}
        clean_messages.append({k: v for k, v in md.items() if k in allowed_message_fields and v is not None})

    # Build final model base URL with tracing metadata
    final_model_base_url = model_base_url
    if model_base_url and ("tracing.fireworks.ai" in model_base_url or model_base_url.startswith("http://localhost")):
        final_model_base_url = build_fireworks_tracing_url(model_base_url, meta, completion_params_base_url)

    # Extract API key from environment or completion_params
    api_key = os.environ.get("FIREWORKS_API_KEY")

    return InitRequest(
        completion_params=completion_params_dict,
        messages=clean_messages,
        tools=row.tools,
        metadata=meta,
        model_base_url=final_model_base_url,
        api_key=api_key,
    )


def update_row_with_remote_trace(
    row: EvaluationRow, output_data_loader: Callable[[DataLoaderConfig], DynamicDataLoader], model_base_url: str
) -> None:
    """Update row with remote trace data using output_data_loader (shared logic)."""
    if not row.execution_metadata.rollout_id:
        return None

    loader_config = DataLoaderConfig(rollout_id=row.execution_metadata.rollout_id, model_base_url=model_base_url)
    data_loader = output_data_loader(loader_config)
    results = data_loader.load()
    output_rows: List[EvaluationRow] = [r for result in results for r in result.rows]

    if len(output_rows) == 0:  # Fallback to original row if no remote data found
        row.rollout_status = Status.rollout_not_found_error("No remote data found for rollout")
        return None
    elif len(output_rows) == 1:  # Return the remote row
        remote_row = output_rows[0]

        # if the remote_row has the same number of messages as the original row, something went wrong
        if len(remote_row.messages) == len(row.messages):
            row.rollout_status = Status.rollout_internal_error(
                "Rollout finished with the same number of messages as the original row"
            )
            return None

        row.messages = remote_row.messages
        row.tools = remote_row.tools
        row.input_metadata.session_data = remote_row.input_metadata.session_data
        row.execution_metadata = remote_row.execution_metadata
        return None
    else:
        raise ValueError("Output data loader should return exactly one row.")
