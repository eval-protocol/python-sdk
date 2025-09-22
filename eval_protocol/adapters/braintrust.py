"""Braintrust adapter for Eval Protocol.

This adapter allows pulling data from Braintrust deployments and converting it
to EvaluationRow format for use in evaluation pipelines.
"""

import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

import requests

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from .base import BaseAdapter
from .utils import extract_messages_from_data


logger = logging.getLogger(__name__)


class TraceConverter(Protocol):
    """Protocol for custom trace-to-EvaluationRow converter functions.

    A converter function should take a Braintrust trace along with processing
    options and return an EvaluationRow or None to skip the trace.
    """

    def __call__(
        self,
        trace: Dict[str, Any],
        include_tool_calls: bool,
    ) -> Optional[EvaluationRow]:
        """Convert a Braintrust trace to an EvaluationRow.

        Args:
            trace: The Braintrust trace object to convert
            include_tool_calls: Whether to include tool calling information

        Returns:
            EvaluationRow or None if the trace should be skipped
        """
        ...


def convert_trace_to_evaluation_row(trace: Dict[str, Any], include_tool_calls: bool = True) -> Optional[EvaluationRow]:
    """Convert a Braintrust trace to EvaluationRow format.

    Args:
        trace: Braintrust trace object
        include_tool_calls: Whether to include tool calling information

    Returns:
        EvaluationRow or None if conversion fails
    """
    try:
        # Extract messages from the trace
        messages = extract_messages_from_trace(trace, include_tool_calls)

        # Extract tools if available
        tools = None
        if include_tool_calls:
            metadata = trace.get("metadata", {})
            tools = metadata.get("tools")
            if not tools:
                hidden_params = metadata.get("hidden_params", {})
                optional_params = hidden_params.get("optional_params", {})
                tools = optional_params.get("tools")

        if not messages:
            return None

        return EvaluationRow(
            messages=messages,
            tools=tools,
            input_metadata=InputMetadata(
                session_data={
                    "braintrust_trace_id": trace.get("id"),
                }
            ),
        )

    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error converting trace %s: %s", trace.get("id", "unknown"), e)
        return None


def extract_messages_from_trace(trace: Dict[str, Any], include_tool_calls: bool = True) -> List[Message]:
    """Extract messages from Braintrust trace input and output.

    Args:
        trace: Braintrust trace object
        include_tool_calls: Whether to include tool calling information

    Returns:
        List of Message objects
    """
    messages = []

    try:
        # Look for complete conversations (input + output arrays)
        input_data = trace.get("input")

        output_data = None
        output_list = trace.get("output", [])
        if output_list and len(output_list) > 0:
            first_output = output_list[0]
            if isinstance(first_output, dict):
                output_data = first_output.get("message")

        # Skip spans without meaningful conversation data
        if not input_data or not output_data:
            return messages

        # Extract messages from input and output
        if input_data:
            messages.extend(extract_messages_from_data(input_data, include_tool_calls))
        if output_data:
            messages.extend(extract_messages_from_data(output_data, include_tool_calls))

    except (AttributeError, ValueError, KeyError) as e:
        logger.warning("Error processing trace %s: %s", trace.get("id", "unknown"), e)

    return messages


class BraintrustAdapter(BaseAdapter):
    """Adapter to pull data from Braintrust and convert to EvaluationRow format.

    This adapter can pull both chat conversations and tool calling traces from
    Braintrust deployments and convert them into the EvaluationRow format expected
    by the evaluation protocol.

    Examples:
        Basic usage:
        >>> adapter = BraintrustAdapter(
        ...     api_key="your_api_key",
        ...     project_id="your_project_id"
        ... )
        >>> btql_query = "select: * from: project_logs('your_project_id') traces limit: 10"
        >>> rows = adapter.get_evaluation_rows(btql_query)

        Using BTQL for custom queries:
        >>> btql_query = '''
        ... select: *
        ... from: project_logs('your_project_id') traces
        ... filter: metadata.agent_name = 'agent_instance'
        ... limit: 50
        ... '''
        >>> rows = adapter.get_evaluation_rows(btql_query)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initialize the Braintrust adapter.

        Args:
            api_key: Braintrust API key (defaults to BRAINTRUST_API_KEY env var)
            api_url: Braintrust API URL (defaults to BRAINTRUST_API_URL env var)
            project_id: Project ID to fetch logs from (defaults to BRAINTRUST_PROJECT_ID env var)
        """
        self.api_key = api_key or os.getenv("BRAINTRUST_API_KEY")
        self.api_url = api_url or os.getenv("BRAINTRUST_API_URL", "https://api.braintrust.dev")
        self.project_id = project_id or os.getenv("BRAINTRUST_PROJECT_ID")

        if not self.api_key:
            raise ValueError("BRAINTRUST_API_KEY environment variable or api_key parameter required")
        if not self.project_id:
            raise ValueError("BRAINTRUST_PROJECT_ID environment variable or project_id parameter required")

    def get_evaluation_rows(
        self,
        btql_query: str,
        include_tool_calls: bool = True,
        converter: Optional[TraceConverter] = None,
    ) -> List[EvaluationRow]:
        """Get evaluation rows using a custom BTQL query.

        Args:
            btql_query: The BTQL query string to execute
            include_tool_calls: Whether to include tool calling information
            converter: Optional custom converter implementing TraceConverter protocol

        Returns:
            List[EvaluationRow]: Converted evaluation rows
        """
        eval_rows = []

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        response = requests.post(f"{self.api_url}/btql", headers=headers, json={"query": btql_query, "fmt": "json"})
        response.raise_for_status()
        query_response = response.json()

        if not query_response or not query_response.get("data"):
            logger.debug("No data returned from BTQL query")
            return eval_rows

        all_traces = query_response["data"]
        logger.debug("BTQL query returned %d traces", len(all_traces))

        # Process each selected trace
        for trace in all_traces:
            try:
                if converter:
                    eval_row = converter(trace, include_tool_calls)
                else:
                    eval_row = convert_trace_to_evaluation_row(trace, include_tool_calls)
                if eval_row:
                    eval_rows.append(eval_row)
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to convert trace %s: %s", trace.get("id", "unknown"), e)
                continue

        logger.info("Successfully processed %d BTQL results into %d evaluation rows", len(all_traces), len(eval_rows))
        return eval_rows

    def upload_scores(self, rows: List[EvaluationRow], model_name: str, mean_score: float) -> None:
        """Upload evaluation scores back to Braintrust traces for tracking and analysis.

        Creates score entries in Braintrust for each unique trace_id found in the evaluation
        rows' session data. This allows you to see evaluation results directly in the
        Braintrust UI alongside the original traces.

        Args:
            rows: List of EvaluationRow objects with session_data containing trace IDs
            model_name: Name of the model (used as the score name in Braintrust)
            mean_score: The calculated mean score to push to Braintrust

        Note:
            Silently handles errors if rows lack session data
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            feedback_items = []
            for trace_id in set(
                row.input_metadata.session_data["braintrust_trace_id"]
                for row in rows
                if row.evaluation_result and row.input_metadata and row.input_metadata.session_data
            ):
                if trace_id:
                    feedback_items.append({"id": trace_id, "scores": {model_name: mean_score}})

            if feedback_items:
                payload = {"feedback": feedback_items}

                response = requests.post(
                    f"{self.api_url}/v1/project_logs/{self.project_id}/feedback",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

        except Exception as e:
            logger.warning("Failed to push scores to Braintrust: %s", e)

    def upload_score(self, row: EvaluationRow, model_name: str) -> None:
        """Upload evaluation score for a single row back to Braintrust.

        Args:
            row: Single EvaluationRow with evaluation_result and session_data containing trace ID
            model_name: Name of the model (used as the score name in Braintrust)
        """
        try:
            if (
                row.evaluation_result
                and row.evaluation_result.is_score_valid
                and row.input_metadata
                and row.input_metadata.session_data
                and "braintrust_trace_id" in row.input_metadata.session_data
            ):
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                trace_id = row.input_metadata.session_data["braintrust_trace_id"]
                if trace_id:
                    feedback_items = [{"id": trace_id, "scores": {model_name: row.evaluation_result.score}}]

                    response = requests.post(
                        f"{self.api_url}/v1/feedback",
                        headers=headers,
                        json={"feedback": feedback_items},
                        timeout=30,
                    )
                    response.raise_for_status()
        except Exception as e:
            logger.warning("Failed to upload single score to Braintrust: %s", e)


def create_braintrust_adapter(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    project_id: Optional[str] = None,
) -> BraintrustAdapter:
    """Factory function to create a Braintrust adapter."""
    return BraintrustAdapter(
        api_key=api_key,
        api_url=api_url,
        project_id=project_id,
    )


__all__ = ["BraintrustAdapter", "create_braintrust_adapter"]
