"""Langfuse adapter for Eval Protocol.

This adapter allows pulling data from Langfuse deployments and converting it
to EvaluationRow format for use in evaluation pipelines.
"""

from __future__ import annotations
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, cast

from langfuse.api.resources.commons.types.observations_view import ObservationsView
from eval_protocol.models import EvaluationRow, InputMetadata, ExecutionMetadata, Message
from .base import BaseAdapter
from .utils import extract_messages_from_data

logger = logging.getLogger(__name__)


class TraceConverter(Protocol):
    """Protocol for custom trace-to-EvaluationRow converter functions.

    A converter function should take a Langfuse trace along with processing
    options and return an EvaluationRow or None to skip the trace.
    """

    def __call__(
        self,
        trace: "TraceWithFullDetails",
        include_tool_calls: bool,
        span_name: Optional[str],
    ) -> Optional[EvaluationRow]:
        """Convert a Langfuse trace to an EvaluationRow.

        Args:
            trace: The Langfuse trace object to convert
            include_tool_calls: Whether to include tool calling information
            span_name: Optional span name to extract messages from

        Returns:
            EvaluationRow or None if the trace should be skipped
        """
        ...


try:
    from langfuse import get_client  # pyright: ignore[reportPrivateImportUsage]

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

if TYPE_CHECKING:
    from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
    from langfuse.api.resources.commons.types.observations_view import ObservationsView


def convert_trace_to_evaluation_row(
    trace: "TraceWithFullDetails", include_tool_calls: bool = True, span_name: Optional[str] = None
) -> Optional[EvaluationRow]:
    """Convert a Langfuse trace to EvaluationRow format.

    Args:
        trace: Langfuse trace object
        include_tool_calls: Whether to include tool calling information
        span_name: If provided, extract messages from generations within this named span

    Returns:
        EvaluationRow or None if conversion fails
    """
    try:
        # Extract messages from trace input and output
        messages = extract_messages_from_trace(trace, include_tool_calls, span_name)

        # Extract tools if available
        tools = None
        if include_tool_calls and isinstance(trace.input, dict) and "tools" in trace.input:
            tools = trace.input["tools"]

        if not messages:
            return None

        execution_metadata = ExecutionMetadata()
        row_id = None

        if trace.tags:
            for tag in trace.tags:
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
                    "langfuse_trace_id": trace.id,  # Store the trace ID here
                },
            ),
            execution_metadata=execution_metadata,
        )

    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error converting trace %s: %s", trace.id, e)
        return None


def extract_messages_from_trace(
    trace: "TraceWithFullDetails", include_tool_calls: bool = True, span_name: Optional[str] = None
) -> List[Message]:
    """Extract messages from Langfuse trace input and output.

    Args:
        trace: Langfuse trace object
        include_tool_calls: Whether to include tool calling information
        span_name: If provided, extract messages from generations within this named span

    Returns:
        List of Message objects
    """
    messages = []

    if span_name:  # Look for a generation tied to a span name
        try:
            # Find the final generation in the named span
            gen: "ObservationsView | None" = get_final_generation_in_span(trace, span_name)
            if not gen:
                return messages

            # Extract messages from generation input and output
            if gen.input:
                messages.extend(extract_messages_from_data(gen.input, include_tool_calls))
            if gen.output:
                messages.extend(extract_messages_from_data(gen.output, include_tool_calls))

            return messages

        except Exception as e:
            logger.error("Failed to extract messages from span '%s' in trace %s: %s", span_name, trace.id, e)
            return messages

    else:
        try:
            # Extract messages from trace input and output
            if trace.input:
                messages.extend(extract_messages_from_data(trace.input, include_tool_calls))
            if trace.output:
                messages.extend(extract_messages_from_data(trace.output, include_tool_calls))
        except (AttributeError, ValueError, KeyError) as e:
            logger.warning("Error processing trace %s: %s", trace.id, e)

        # Fallback: use the last GENERATION observation which typically contains full chat history
        if not messages:
            try:
                all_observations = getattr(trace, "observations", None) or []
                gens: List[ObservationsView] = [
                    obs for obs in all_observations if getattr(obs, "type", None) == "GENERATION"
                ]
                if gens:
                    gens.sort(key=lambda x: x.start_time)
                    last_gen = gens[-1]
                    if getattr(last_gen, "input", None):
                        messages.extend(extract_messages_from_data(getattr(last_gen, "input"), include_tool_calls))
                    if getattr(last_gen, "output", None):
                        messages.extend(extract_messages_from_data(getattr(last_gen, "output"), include_tool_calls))
            except Exception as e:
                logger.warning("Failed to extract from last generation for trace %s: %s", trace.id, e)

    return messages


def get_final_generation_in_span(trace: "TraceWithFullDetails", span_name: str) -> "ObservationsView | None":
    """Get the final generation within a named span that contains full message history.

    Args:
        trace: Langfuse trace object
        span_name: Name of the span to search for

    Returns:
        The final generation object, or None if not found
    """
    # Get all observations from the trace
    all_observations = trace.observations

    # Find a span with the given name that has generation children
    parent_span = None
    for obs in all_observations:
        if obs.name == span_name and obs.type == "SPAN":
            # Check if this span has generation children
            has_generations = any(
                child.type == "GENERATION" and child.parent_observation_id == obs.id for child in all_observations
            )
            if has_generations:
                parent_span = obs
                break

    if not parent_span:
        logger.warning("No span named '%s' found in trace %s", span_name, trace.id)
        return None

    # Find all generations within this span
    generations: List["ObservationsView"] = []
    for obs in all_observations:
        if obs.type == "GENERATION" and obs.parent_observation_id == parent_span.id:
            generations.append(obs)

    if not generations:
        logger.warning("No generations found in span '%s' in trace %s", span_name, trace.id)
        return None

    # Sort generations by start time for chronological order
    generations.sort(key=lambda x: x.start_time)

    # Return the final generation (contains full message history)
    return generations[-1]


class LangfuseAdapter(BaseAdapter):
    """Adapter to pull data from Langfuse and convert to EvaluationRow format.

    This adapter can pull both chat conversations and tool calling traces from
    Langfuse deployments and convert them into the EvaluationRow format expected
    by the evaluation protocol.

    Examples:
        Basic usage:
        >>> adapter = LangfuseAdapter(
        ...     public_key="your_public_key",
        ...     secret_key="your_secret_key",
        ...     host="https://your-langfuse-deployment.com"
        ... )
        >>> rows = list(adapter.get_evaluation_rows(limit=10))

        Filter by specific criteria:
        >>> rows = list(adapter.get_evaluation_rows(
        ...     limit=50,
        ...     tags=["production"],
        ...     user_id="specific_user",
        ...     from_timestamp=datetime.now() - timedelta(days=7)
        ... ))
    """

    def __init__(self, client: Optional[Any] = None):
        """Initialize the Langfuse adapter."""
        if not LANGFUSE_AVAILABLE:
            raise ImportError("Langfuse not installed. Install with: pip install 'eval-protocol[langfuse]'")

        self.client = client or cast(Any, get_client)()

    def get_evaluation_rows(
        self,
        limit: int = 100,
        sample_size: Optional[int] = None,
        tags: Optional[List[str]] = None,
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
        sleep_between_gets: float = 2.5,
        max_retries: int = 3,
        span_name: Optional[str] = None,
        converter: Optional[TraceConverter] = None,
    ) -> List[EvaluationRow]:
        """Pull traces from Langfuse and convert to EvaluationRow format.

        Args:
            limit: Max number of trace summaries to collect via pagination (pre-sampling)
            sample_size: Optional number of traces to randomly sample from collected summaries (if None, process all)
            tags: Filter by specific tags
            user_id: Filter by user ID
            session_id: Filter by session ID
            name: Filter by trace name
            environment: Filter by environment (e.g., production, staging, development)
            version: Filter by trace version
            release: Filter by trace release
            fields: Comma-separated list of fields to include (e.g., 'core,scores,metrics')
            hours_back: Filter traces from this many hours ago
            from_timestamp: Explicit start time (overrides hours_back)
            to_timestamp: Explicit end time (overrides hours_back)
            include_tool_calls: Whether to include tool calling traces
            sleep_between_gets: Sleep time between individual trace.get() calls (2.5s for 30 req/min limit)
            max_retries: Maximum retries for rate limit errors
            span_name: If provided, extract messages from generations within this named span
            converter: Optional custom converter implementing TraceConverter protocol.
                If provided, this will be used instead of the default conversion logic.

        Returns:
            List[EvaluationRow]: Converted evaluation rows
        """
        eval_rows = []

        # Determine time window: explicit from/to takes precedence over hours_back
        if from_timestamp is None and to_timestamp is None and hours_back:
            to_timestamp = datetime.now()
            from_timestamp = to_timestamp - timedelta(hours=hours_back)

        # Collect trace summaries via pagination (up to limit)
        all_traces = []
        page = 1
        collected = 0

        while collected < limit:
            current_page_limit = min(100, limit - collected)  # Langfuse API max is 100

            logger.debug(
                "Fetching page %d with limit %d (collected: %d/%d)", page, current_page_limit, collected, limit
            )

            # Fetch trace list with retry logic
            traces = None
            list_retries = 0
            while list_retries < max_retries:
                try:
                    traces = self.client.api.trace.list(
                        page=page,
                        limit=current_page_limit,
                        tags=tags,
                        user_id=user_id,
                        session_id=session_id,
                        name=name,
                        environment=environment,
                        version=version,
                        release=release,
                        fields=fields,
                        from_timestamp=from_timestamp,
                        to_timestamp=to_timestamp,
                        order_by="timestamp.desc",
                    )

                    # If no results, possible due to indexing delay--remote rollout processor just finished pushing rows to Langfuse
                    if traces and traces.meta and traces.meta.total_items == 0 and page == 1:
                        raise Exception("Empty results")

                    break
                except Exception as e:
                    list_retries += 1
                    if list_retries < max_retries and ("429" in str(e) or "Empty results" in str(e)):
                        sleep_time = 2**list_retries  # Exponential backoff
                        logger.warning(
                            "Retrying in %ds (attempt %d/%d): %s", sleep_time, list_retries, max_retries, str(e)
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error("Failed to fetch trace list after %d retries: %s", max_retries, e)
                        return eval_rows  # Return what we have so far

            if not traces or not traces.data:
                logger.debug("No more traces found on page %d", page)
                break

            logger.debug("Collected %d traces from page %d", len(traces.data), page)

            all_traces.extend(traces.data)
            collected += len(traces.data)

            # Check if we have more pages
            if hasattr(traces.meta, "page") and hasattr(traces.meta, "total_pages"):
                if traces.meta.page >= traces.meta.total_pages:
                    break
            elif len(traces.data) < current_page_limit:
                break

            page += 1

        if not all_traces:
            logger.debug("No traces found")
            return eval_rows

        # Optionally sample traces to fetch full details (respect rate limits)
        if sample_size is not None:
            actual_sample_size = min(sample_size, len(all_traces))
            selected_traces = random.sample(all_traces, actual_sample_size)
            logger.debug("Randomly selected %d traces from %d collected", actual_sample_size, len(all_traces))
        else:
            selected_traces = all_traces
            logger.debug("Processing all %d collected traces (no sampling)", len(all_traces))

        # Process each selected trace with sleep and retry logic
        for trace_info in selected_traces:
            # Sleep between gets to avoid rate limits
            if sleep_between_gets > 0:
                time.sleep(sleep_between_gets)

            # Fetch full trace details with retry logic
            trace_full = None
            detail_retries = 0
            while detail_retries < max_retries:
                try:
                    # Some SDKs don't support fields= on get; call without it
                    trace_full = self.client.api.trace.get(trace_info.id)
                    break
                except Exception as e:
                    detail_retries += 1
                    if "429" in str(e) and detail_retries < max_retries:
                        sleep_time = 2**detail_retries  # Exponential backoff
                        logger.warning(
                            "Rate limit hit on trace.get(%s), retrying in %ds (attempt %d/%d)",
                            trace_info.id,
                            sleep_time,
                            detail_retries,
                            max_retries,
                        )
                        time.sleep(sleep_time)
                    elif "Not Found" in str(e) or "404" in str(e):
                        # Skip missing traces quickly
                        logger.debug("Trace %s not found, skipping", trace_info.id)
                        trace_full = None
                        break
                    else:
                        logger.warning("Failed to fetch trace %s after %d retries: %s", trace_info.id, max_retries, e)
                        break  # Skip this trace

            if trace_full:
                try:
                    if converter:
                        eval_row = converter(trace_full, include_tool_calls, span_name)
                    else:
                        eval_row = convert_trace_to_evaluation_row(trace_full, include_tool_calls, span_name)
                    if eval_row:
                        eval_rows.append(eval_row)
                except (AttributeError, ValueError, KeyError) as e:
                    logger.warning("Failed to convert trace %s: %s", trace_info.id, e)
                    continue

        logger.info(
            "Successfully processed %d selected traces into %d evaluation rows", len(selected_traces), len(eval_rows)
        )
        return eval_rows

    def get_evaluation_rows_by_ids(
        self,
        trace_ids: List[str],
        include_tool_calls: bool = True,
        span_name: Optional[str] = None,
        converter: Optional[TraceConverter] = None,
    ) -> List[EvaluationRow]:
        """Get specific traces by their IDs and convert to EvaluationRow format.

        Args:
            trace_ids: List of trace IDs to fetch
            include_tool_calls: Whether to include tool calling traces
            span_name: If provided, extract messages from generations within this named span
            converter: Optional custom converter implementing TraceConverter protocol.
                If provided, this will be used instead of the default conversion logic.

        Returns:
            List[EvaluationRow]: Converted evaluation rows
        """
        eval_rows = []
        for trace_id in trace_ids:
            try:
                trace: TraceWithFullDetails = self.client.api.trace.get(trace_id)
                if converter:
                    eval_row = converter(trace, include_tool_calls, span_name)
                else:
                    eval_row = convert_trace_to_evaluation_row(trace, include_tool_calls, span_name)
                if eval_row:
                    eval_rows.append(eval_row)
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to fetch/convert trace %s: %s", trace_id, e)
                continue
        return eval_rows

    def upload_scores(self, rows: List[EvaluationRow], model_name: str, mean_score: float) -> None:
        """Upload evaluation scores back to Langfuse traces for tracking and analysis.

        Creates a score entry in Langfuse for each unique trace_id found in the evaluation
        rows' session data. This allows you to see evaluation results directly in the
        Langfuse UI alongside the original traces.

        Args:
            rows: List of EvaluationRow objects with session_data containing trace IDs
            model_name: Name of the model (used as the score name in Langfuse)
            mean_score: The calculated mean score to push to Langfuse
        """
        try:
            for trace_id in set(
                (row.input_metadata.session_data or {}).get("langfuse_trace_id")
                for row in rows
                if row.input_metadata and row.input_metadata.session_data
            ):
                if trace_id:
                    try:
                        self.client.api.score.create(
                            trace_id=trace_id,
                            name=model_name,
                            value=mean_score,
                        )
                    except Exception:
                        # Fallback to legacy client if available in some environments
                        create_score = getattr(self.client, "create_score", None)
                        if callable(create_score):
                            create_score(trace_id=trace_id, name=model_name, value=mean_score)
        except Exception as e:
            logger.warning("Failed to push scores to Langfuse: %s", e)

    def upload_score(self, row: EvaluationRow, model_name: str) -> None:
        """Upload evaluation score for a single row back to Langfuse.

        Args:
            row: Single EvaluationRow with evaluation_result and session_data containing trace ID
            model_name: Name of the model (used as the score name in Langfuse)
        """
        try:
            if (
                row.evaluation_result
                and row.evaluation_result.is_score_valid
                and row.input_metadata
                and row.input_metadata.session_data
                and "langfuse_trace_id" in row.input_metadata.session_data
            ):
                trace_id = row.input_metadata.session_data["langfuse_trace_id"]
                if trace_id:
                    self.client.create_score(
                        trace_id=trace_id,
                        name=model_name,
                        value=row.evaluation_result.score,
                    )
        except Exception as e:
            logger.warning("Failed to push score to Langfuse: %s", e)


def create_langfuse_adapter() -> LangfuseAdapter:
    """Factory function to create a Langfuse adapter."""

    return LangfuseAdapter()
