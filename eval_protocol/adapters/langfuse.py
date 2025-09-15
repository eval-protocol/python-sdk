"""Langfuse adapter for Eval Protocol.

This adapter allows pulling data from Langfuse deployments and converting it
to EvaluationRow format for use in evaluation pipelines.
"""

from langfuse.api.resources.commons.types.observations_view import ObservationsView
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, cast

from eval_protocol.models import EvaluationRow, InputMetadata, Message

logger = logging.getLogger(__name__)

try:
    from langfuse import get_client  # pyright: ignore[reportPrivateImportUsage]
    from langfuse.api.resources.trace.types.traces import Traces
    from langfuse.api.resources.commons.types.trace import Trace
    from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class LangfuseAdapter:
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

    def __init__(self):
        """Initialize the Langfuse adapter."""
        if not LANGFUSE_AVAILABLE:
            raise ImportError("Langfuse not installed. Install with: pip install 'eval-protocol[langfuse]'")

        self.client = get_client()

    def get_evaluation_rows(
        self,
        limit: int = 100,
        sample_size: int = 50,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hours_back: Optional[int] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        include_tool_calls: bool = True,
        sleep_between_gets: float = 2.5,
        max_retries: int = 3,
    ) -> List[EvaluationRow]:
        """Pull traces from Langfuse and convert to EvaluationRow format.

        Args:
            limit: Max number of trace summaries to collect via pagination (pre-sampling)
            sample_size: Number of traces to fetch full details for (sampled from collected summaries)
            tags: Filter by specific tags
            user_id: Filter by user ID
            session_id: Filter by session ID
            hours_back: Filter traces from this many hours ago
            from_timestamp: Explicit start time (overrides hours_back)
            to_timestamp: Explicit end time (overrides hours_back)
            include_tool_calls: Whether to include tool calling traces
            sleep_between_gets: Sleep time between individual trace.get() calls (2.5s for 30 req/min limit)
            max_retries: Maximum retries for rate limit errors

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
                        from_timestamp=from_timestamp,
                        to_timestamp=to_timestamp,
                        order_by="timestamp.desc",
                    )
                    break
                except Exception as e:
                    list_retries += 1
                    if "429" in str(e) and list_retries < max_retries:
                        sleep_time = 2**list_retries  # Exponential backoff
                        logger.warning(
                            "Rate limit hit on trace.list(), retrying in %ds (attempt %d/%d)",
                            sleep_time,
                            list_retries,
                            max_retries,
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

        # Randomly sample traces to fetch full details (respect rate limits)
        actual_sample_size = min(sample_size, len(all_traces))
        selected_traces = random.sample(all_traces, actual_sample_size)

        logger.debug("Randomly selected %d traces from %d collected", actual_sample_size, len(all_traces))

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
                    else:
                        logger.warning("Failed to fetch trace %s after %d retries: %s", trace_info.id, max_retries, e)
                        break  # Skip this trace

            if trace_full:
                try:
                    eval_row = self._convert_trace_to_evaluation_row(trace_full, include_tool_calls)
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
    ) -> List[EvaluationRow]:
        """Get specific traces by their IDs and convert to EvaluationRow format.

        Args:
            trace_ids: List of trace IDs to fetch
            include_tool_calls: Whether to include tool calling traces

        Yields:
            EvaluationRow: Converted evaluation rows
        """
        eval_rows = []
        for trace_id in trace_ids:
            try:
                trace: TraceWithFullDetails = self.client.api.trace.get(trace_id)
                eval_row = self._convert_trace_to_evaluation_row(trace, include_tool_calls)
                if eval_row:
                    eval_rows.append(eval_row)
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to fetch/convert trace %s: %s", trace_id, e)
                continue
        return eval_rows

    def _convert_trace_to_evaluation_row(
        self, trace: TraceWithFullDetails, include_tool_calls: bool = True
    ) -> Optional[EvaluationRow]:
        """Convert a Langfuse trace to EvaluationRow format.

        Args:
            trace: Langfuse trace object
            include_tool_calls: Whether to include tool calling information

        Returns:
            EvaluationRow or None if conversion fails
        """
        try:
            # Extract messages from trace input and output
            messages = self._extract_messages_from_trace(trace, include_tool_calls)

            # Extract tools if available
            tools = None
            if include_tool_calls and isinstance(trace.input, dict) and "tools" in trace.input:
                tools = trace.input["tools"]

            if not messages:
                return None

            return EvaluationRow(
                messages=messages,
                tools=tools,
                input_metadata=InputMetadata(
                    session_data={
                        "langfuse_trace_id": trace.id,  # Store the trace ID here
                    }
                ),
            )

        except (AttributeError, ValueError, KeyError) as e:
            logger.error("Error converting trace %s: %s", trace.id, e)
            return None

    def _extract_messages_from_trace(
        self, trace: TraceWithFullDetails, include_tool_calls: bool = True
    ) -> List[Message]:
        """Extract messages from Langfuse trace input and output.

        Args:
            trace: Langfuse trace object
            include_tool_calls: Whether to include tool calling information

        Returns:
            List of Message objects
        """
        messages = []

        try:
            # Handle trace input
            if hasattr(trace, "input") and trace.input:
                if isinstance(trace.input, dict):
                    if "messages" in trace.input:
                        # OpenAI-style messages format
                        for msg in trace.input["messages"]:
                            messages.append(self._dict_to_message(msg, include_tool_calls))
                    elif "role" in trace.input:
                        # Single message format
                        messages.append(self._dict_to_message(trace.input, include_tool_calls))
                    elif "prompt" in trace.input:
                        # Simple prompt format
                        messages.append(Message(role="user", content=str(trace.input["prompt"])))
                elif isinstance(trace.input, list):
                    # Direct list of message dicts
                    for msg in trace.input:
                        messages.append(self._dict_to_message(msg, include_tool_calls))
                elif isinstance(trace.input, str):
                    # Simple string input
                    messages.append(Message(role="user", content=trace.input))

            # Handle trace output
            if hasattr(trace, "output") and trace.output:
                if isinstance(trace.output, dict):
                    if "content" in trace.output:
                        messages.append(Message(role="assistant", content=str(trace.output["content"])))
                    elif "message" in trace.output:
                        msg_dict = trace.output["message"]
                        messages.append(self._dict_to_message(msg_dict, include_tool_calls))
                    else:
                        # Fallback: convert entire output to string
                        messages.append(Message(role="assistant", content=str(trace.output)))
                elif isinstance(trace.output, list):
                    # Direct list of message dicts (same as input handling)
                    for msg in trace.output:
                        messages.append(self._dict_to_message(msg, include_tool_calls))
                elif isinstance(trace.output, str):
                    messages.append(Message(role="assistant", content=trace.output))

        except (AttributeError, ValueError, KeyError) as e:
            logger.warning("Error processing trace %s: %s", trace.id, e)

        return messages

    def _dict_to_message(self, msg_dict: Dict[str, Any], include_tool_calls: bool = True) -> Message:
        """Convert a dictionary to a Message object.

        Args:
            msg_dict: Dictionary containing message data
            include_tool_calls: Whether to include tool calling information

        Returns:
            Message object
        """
        # Extract basic message components
        role = msg_dict.get("role", "assistant")
        content = msg_dict.get("content")
        name = msg_dict.get("name")

        # Handle tool calls if enabled
        tool_calls = None
        tool_call_id = None
        function_call = None

        if include_tool_calls:
            if "tool_calls" in msg_dict:
                tool_calls = msg_dict["tool_calls"]
            if "tool_call_id" in msg_dict:
                tool_call_id = msg_dict["tool_call_id"]
            if "function_call" in msg_dict:
                function_call = msg_dict["function_call"]

        return Message(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            function_call=function_call,
        )


def create_langfuse_adapter() -> LangfuseAdapter:
    """Factory function to create a Langfuse adapter."""

    return LangfuseAdapter()
