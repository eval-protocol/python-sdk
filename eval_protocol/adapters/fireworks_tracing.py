"""Fireworks Tracing adapter for Eval Protocol.

This adapter uses the Fireworks tracing proxy at tracing.fireworks.ai
to pull data from Langfuse deployments with simplified retry logic handling.
"""

from __future__ import annotations
import logging
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
import os

from eval_protocol.models import EvaluationRow, InputMetadata, ExecutionMetadata, Message
from .base import BaseAdapter
from .utils import extract_messages_from_data

logger = logging.getLogger(__name__)


class TraceDictConverter(Protocol):
    """Protocol for custom trace dictionary-to-EvaluationRow converter functions.

    A converter function should take a trace dictionary along with processing
    options and return an EvaluationRow or None to skip the trace.
    """

    def __call__(
        self,
        trace: Dict[str, Any],
        include_tool_calls: bool,
        span_name: Optional[str],
    ) -> Optional[EvaluationRow]:
        """Convert a trace dictionary to an EvaluationRow.

        Args:
            trace: The trace dictionary to convert
            include_tool_calls: Whether to include tool calling information
            span_name: Optional span name to extract messages from

        Returns:
            EvaluationRow or None if the trace should be skipped
        """
        ...


def convert_trace_dict_to_evaluation_row(
    trace: Dict[str, Any], include_tool_calls: bool = True, span_name: Optional[str] = None
) -> Optional[EvaluationRow]:
    """Convert a trace dictionary (from proxy API) to EvaluationRow format.

    Args:
        trace: Trace dictionary from Fireworks proxy API
        include_tool_calls: Whether to include tool calling information
        span_name: If provided, extract messages from generations within this named span

    Returns:
        EvaluationRow or None if conversion fails
    """
    try:
        # Extract messages from trace input and output
        messages = extract_messages_from_trace_dict(trace, include_tool_calls, span_name)

        # Extract tools if available
        tools = None
        if include_tool_calls and isinstance(trace.get("input"), dict) and "tools" in trace["input"]:
            tools = trace["input"]["tools"]

        if not messages:
            return None

        execution_metadata = ExecutionMetadata()
        row_id = None

        # Extract metadata from tags
        tags = trace.get("tags", [])
        if tags:
            for tag in tags:
                if tag.startswith("invocation_id:"):
                    execution_metadata.invocation_id = tag.split(":", 1)[1]
                elif tag.startswith("experiment_id:"):
                    execution_metadata.experiment_id = tag.split(":", 1)[1]
                elif tag.startswith("rollout_id:"):
                    execution_metadata.rollout_id = tag.split(":", 1)[1]
                elif tag.startswith("run_id:"):
                    execution_metadata.run_id = tag.split(":", 1)[1]
                elif tag.startswith("row_id:"):
                    row_id = tag.split(":", 1)[1]

                if (
                    execution_metadata.invocation_id
                    and execution_metadata.experiment_id
                    and execution_metadata.rollout_id
                    and execution_metadata.run_id
                    and row_id
                ):
                    break  # Break early if we've found all the metadata we need

        return EvaluationRow(
            messages=messages,
            tools=tools,
            input_metadata=InputMetadata(
                row_id=row_id,
                session_data={
                    "langfuse_trace_id": trace.get("id"),  # Store the trace ID here
                },
            ),
            execution_metadata=execution_metadata,
        )

    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error converting trace %s: %s", trace.get("id"), e)
        return None


def extract_messages_from_trace_dict(
    trace: Dict[str, Any], include_tool_calls: bool = True, span_name: Optional[str] = None
) -> List[Message]:
    """Extract messages from trace dictionary.

    Args:
        trace: Trace dictionary from proxy API
        include_tool_calls: Whether to include tool calling information
        span_name: If provided, extract messages from generations within this named span

    Returns:
        List of Message objects
    """
    messages = []

    if span_name:  # Look for a generation tied to a span name
        try:
            # Find the final generation in the named span
            gen = get_final_generation_in_span_dict(trace, span_name)
            if not gen:
                return messages

            # Extract messages from generation input and output
            if gen.get("input"):
                messages.extend(extract_messages_from_data(gen["input"], include_tool_calls))
            if gen.get("output"):
                messages.extend(extract_messages_from_data(gen["output"], include_tool_calls))

            return messages

        except Exception as e:
            logger.error("Failed to extract messages from span '%s' in trace %s: %s", span_name, trace.get("id"), e)
            return messages

    else:
        try:
            # Extract messages from trace input and output
            if trace.get("input"):
                messages.extend(extract_messages_from_data(trace["input"], include_tool_calls))
            if trace.get("output"):
                messages.extend(extract_messages_from_data(trace["output"], include_tool_calls))
        except (AttributeError, ValueError, KeyError) as e:
            logger.warning("Error processing trace %s: %s", trace.get("id"), e)

        # Fallback: use the last GENERATION observation which typically contains full chat history
        if not messages:
            try:
                all_observations = trace.get("observations", [])
                gens = [obs for obs in all_observations if obs.get("type") == "GENERATION"]
                if gens:
                    gens.sort(key=lambda x: x.get("start_time", ""))
                    last_gen = gens[-1]
                    if last_gen.get("input"):
                        messages.extend(extract_messages_from_data(last_gen["input"], include_tool_calls))
                    if last_gen.get("output"):
                        messages.extend(extract_messages_from_data(last_gen["output"], include_tool_calls))
            except Exception as e:
                logger.warning("Failed to extract from last generation for trace %s: %s", trace.get("id"), e)

    return messages


def get_final_generation_in_span_dict(trace: Dict[str, Any], span_name: str) -> Optional[Dict[str, Any]]:
    """Get the final generation within a named span from trace dictionary.

    Args:
        trace: Trace dictionary
        span_name: Name of the span to search for

    Returns:
        The final generation dictionary, or None if not found
    """
    # Get all observations from the trace
    all_observations = trace.get("observations", [])

    # Find a span with the given name that has generation children
    parent_span = None
    for obs in all_observations:
        if obs.get("name") == span_name and obs.get("type") == "SPAN":
            # Check if this span has generation children
            has_generations = any(
                child.get("type") == "GENERATION" and child.get("parent_observation_id") == obs.get("id")
                for child in all_observations
            )
            if has_generations:
                parent_span = obs
                break

    if not parent_span:
        logger.warning("No span named '%s' found in trace %s", span_name, trace.get("id"))
        return None

    # Find all generations within this span
    generations = []
    for obs in all_observations:
        if obs.get("type") == "GENERATION" and obs.get("parent_observation_id") == parent_span.get("id"):
            generations.append(obs)

    if not generations:
        logger.warning("No generations found in span '%s' in trace %s", span_name, trace.get("id"))
        return None

    # Sort generations by start time for chronological order
    generations.sort(key=lambda x: x.get("start_time", ""))

    # Return the final generation (contains full message history)
    return generations[-1]


class FireworksTracingAdapter(BaseAdapter):
    """Adapter to pull data from Langfuse via Fireworks tracing proxy.

    This adapter uses the Fireworks tracing proxy API which handles retry logic
    and rate limiting internally, simplifying the client-side implementation.

    Examples:
        Basic usage (default project):
        >>> adapter = FireworksTracingAdapter()
        >>> rows = list(adapter.get_evaluation_rows(tags=["rollout_id:xyz"], limit=10))

        With explicit project ID:
        >>> adapter = FireworksTracingAdapter(
        ...     project_id="your_project_id",
        ...     base_url="https://tracing.fireworks.ai"
        ... )
        >>> rows = list(adapter.get_evaluation_rows(tags=["production"], limit=10))

        Filter by specific criteria:
        >>> rows = list(adapter.get_evaluation_rows(
        ...     tags=["production"],
        ...     limit=50,
        ...     hours_back=24
        ... ))
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        base_url: str = "https://tracing.fireworks.ai",
        timeout: int = 300,
    ):
        """Initialize the Fireworks Tracing adapter.

        Args:
            project_id: Optional project ID. If not provided, uses the default project configured on the server.
            base_url: The base URL of the tracing proxy (default: https://tracing.fireworks.ai)
            timeout: Request timeout in seconds (default: 300)
        """
        self.project_id = project_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search_logs(self, tags: List[str], limit: int = 100, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch logs from Fireworks tracing gateway /logs endpoint.

        Returns entries with keys: timestamp, message, severity, tags.
        """
        if not tags:
            raise ValueError("At least one tag is required to fetch logs")

        from ..common_utils import get_user_agent

        headers = {
            "Authorization": f"Bearer {os.environ.get('FIREWORKS_API_KEY')}",
            "User-Agent": get_user_agent(),
        }
        params: Dict[str, Any] = {"tags": tags, "limit": limit, "hours_back": hours_back, "program": "eval_protocol"}

        # Try /logs first, fall back to /v1/logs if not found
        urls_to_try = [f"{self.base_url}/logs", f"{self.base_url}/v1/logs"]
        data: Dict[str, Any] = {}
        last_error: Optional[str] = None
        for url in urls_to_try:
            try:
                response = requests.get(url, params=params, timeout=self.timeout, headers=headers)
                if response.status_code == 404:
                    # Try next variant
                    last_error = f"404 for {url}"
                    continue
                response.raise_for_status()
                data = response.json() or {}
                break
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                continue
        else:
            # All attempts failed
            if last_error:
                logger.error("Failed to fetch logs from Fireworks (tried %s): %s", urls_to_try, last_error)
            return []

        entries: List[Dict[str, Any]] = data.get("entries", []) or []
        # Normalize minimal shape
        results: List[Dict[str, Any]] = []
        for e in entries:
            results.append(
                {
                    "timestamp": e.get("timestamp"),
                    "message": e.get("message"),
                    "severity": e.get("severity", "INFO"),
                    "tags": e.get("tags", []),
                    "status": e.get("status"),
                }
            )
        return results

    def get_evaluation_rows(
        self,
        tags: List[str],
        limit: int = 100,
        sample_size: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        environment: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        fields: Optional[str] = None,
        hours_back: Optional[int] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        include_tool_calls: bool = True,
        sleep_between_gets: float = 0.1,
        max_retries: int = 3,
        span_name: Optional[str] = None,
        converter: Optional[TraceDictConverter] = None,
    ) -> List[EvaluationRow]:
        """Pull traces from Langfuse via proxy and convert to EvaluationRow format.

        Args:
            tags: REQUIRED - Filter by specific tags (prevents fetching all traces).
                Must provide at least one tag (e.g., ['rollout_id:xyz'], ['production'])
            limit: Max number of trace summaries to collect via pagination
            sample_size: Optional number of traces to randomly sample (if None, process all)
            user_id: Filter by user ID
            session_id: Filter by session ID
            name: Filter by trace name
            environment: Filter by environment (e.g., production, staging, development)
            version: Filter by trace version
            release: Filter by trace release
            fields: Comma-separated list of fields to include
            hours_back: Filter traces from this many hours ago
            from_timestamp: Explicit start time (ISO format)
            to_timestamp: Explicit end time (ISO format)
            include_tool_calls: Whether to include tool calling traces
            sleep_between_gets: Sleep time between polling attempts (default: 2.5s)
            max_retries: Max retry attempts used by proxy (default: 3)
            converter: Optional custom converter implementing TraceDictConverter protocol.
                If provided, this will be used instead of the default conversion logic.

        Returns:
            List[EvaluationRow]: Converted evaluation rows

        Raises:
            ValueError: If tags list is empty
        """
        # Validate that tags are provided
        if not tags or len(tags) == 0:
            raise ValueError("At least one tag is required to fetch traces")

        eval_rows = []

        # Build query parameters for GET request
        params = {
            "limit": limit,
            "sample_size": sample_size,
            "tags": tags,
            "user_id": user_id,
            "session_id": session_id,
            "name": name,
            "environment": environment,
            "version": version,
            "release": release,
            "fields": fields,
            "hours_back": hours_back,
            "from_timestamp": from_timestamp.isoformat() if from_timestamp else None,
            "to_timestamp": to_timestamp.isoformat() if to_timestamp else None,
            "sleep_between_gets": sleep_between_gets,
            "max_retries": max_retries,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Make request to proxy (using pointwise for efficiency)
        if self.project_id:
            url = f"{self.base_url}/v1/project_id/{self.project_id}/traces/pointwise"
        else:
            url = f"{self.base_url}/v1/traces/pointwise"

        from ..common_utils import get_user_agent

        headers = {
            "Authorization": f"Bearer {os.environ.get('FIREWORKS_API_KEY')}",
            "User-Agent": get_user_agent(),
        }

        result = None
        try:
            response = requests.get(url, params=params, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)

            # Try to extract detail message from response
            if e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", {})
                    error_msg = error_detail or e.response.text
                except Exception:  # In case e.response.json() fails
                    error_msg = f"Proxy error: {e.response.text}"

            logger.error("Failed to fetch traces from proxy (HTTP %s): %s", e.response.status_code, error_msg)
            return eval_rows
        except requests.exceptions.RequestException as e:
            # Non-HTTP errors (network issues, timeouts, etc.)
            logger.error("Failed to fetch traces from proxy: %s", str(e))
            return eval_rows

        # Extract traces from response
        traces = result.get("traces", [])

        # Convert each trace to EvaluationRow
        for trace in traces:
            try:
                if converter:
                    eval_row = converter(trace, include_tool_calls, span_name)
                else:
                    eval_row = convert_trace_dict_to_evaluation_row(trace, include_tool_calls, span_name)
                if eval_row:
                    eval_rows.append(eval_row)
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to convert trace %s: %s", trace.get("id"), e)
                continue

        logger.info("Successfully converted %d traces to evaluation rows", len(eval_rows))
        return eval_rows
