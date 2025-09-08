"""Braintrust adapter for Eval Protocol.

This adapter pulls traces from Braintrust projects and converts them
to EvaluationRow format for evaluation pipelines.
"""

import os
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import requests

from eval_protocol.models import EvaluationRow, InputMetadata, Message

# Keep backward compatibility
from ..integrations.braintrust import reward_fn_to_scorer, scorer_to_reward_fn


class BraintrustAdapter:
    """Minimal adapter to pull traces from Braintrust."""

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
            project_id: Project ID to fetch logs from
        """
        self.api_key = api_key or os.getenv("BRAINTRUST_API_KEY")
        self.api_url = api_url or os.getenv("BRAINTRUST_API_URL", "https://api.braintrust.dev")
        self.project_id = project_id

        if not self.api_key:
            raise ValueError("BRAINTRUST_API_KEY environment variable or api_key parameter required")

    def get_evaluation_rows(
        self,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
    ) -> Iterator[EvaluationRow]:
        """Fetch traces from Braintrust and convert to EvaluationRow format."""
        project_id = project_id or self.project_id
        if not project_id:
            raise ValueError("project_id required")

        # Prepare query parameters for GET request
        params = {"limit": 1000}
        if from_timestamp:
            params["from_timestamp"] = int(from_timestamp.timestamp())
        if to_timestamp:
            params["to_timestamp"] = int(to_timestamp.timestamp())

        # Fetch logs from Braintrust using GET endpoint
        headers = {"Authorization": f"Bearer {self.api_key}"}

        url = f"{self.api_url}/v1/project_logs/{project_id}/fetch"

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        logs = response.json()

        # Convert each log to EvaluationRow
        for log in logs.get("events", []):
            if log.get("metadata", {}).get("agent_name") == "agent_instance":
                try:
                    eval_row = self._convert_log_to_evaluation_row(log)
                    if eval_row:
                        yield eval_row
                except Exception as e:
                    print(f"Warning: Failed to convert log {log.get('id', 'unknown')}: {e}")
                    continue

    def _convert_log_to_evaluation_row(self, log: Dict[str, Any]) -> Optional[EvaluationRow]:
        """Convert a Braintrust log to EvaluationRow format."""
        # Extract messages from the log
        messages = self._extract_messages(log)
        if not messages:
            return None

        # Extract metadata (pulling nothing currently)
        input_metadata = InputMetadata(
            row_id=log.get("id"),
            completion_params=log.get("metadata", {}),
            dataset_info={
                "braintrust_log_id": log.get("id"),
                "braintrust_project_id": self.project_id,
                "span_id": log.get("span_id"),
                "trace_id": log.get("root_span_id"),
            },
        )

        # Extract ground truth from metadata
        metadata = log.get("metadata", {})
        ground_truth = metadata.get("ground_truth")

        return EvaluationRow(
            messages=messages,
            input_metadata=input_metadata,
            ground_truth=str(ground_truth) if ground_truth else None,
        )

    def _extract_messages(self, log: Dict[str, Any]) -> List[Message]:
        """Extract conversation messages from a Braintrust log."""
        messages = []

        # Look for complete conversations (input + output arrays)
        input_data = log.get("input")
        output_data = log.get("output")

        # Skip spans without meaningful conversation data
        if not input_data or not output_data:
            return []

        # Extract input messages (usually just user message)
        if isinstance(input_data, list):
            for msg in input_data:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(Message(role=msg["role"], content=str(msg["content"])))

        # Extract output messages (assistant + tool responses)
        if isinstance(output_data, list):
            for msg in output_data:
                if isinstance(msg, dict) and "role" in msg:
                    # Handle tool calls in assistant messages
                    tool_calls = msg.get("tool_calls") if msg["role"] == "assistant" else None
                    tool_call_id = msg.get("tool_call_id") if msg["role"] == "tool" else None
                    name = msg.get("name") if msg["role"] == "tool" else None

                    messages.append(
                        Message(
                            role=msg["role"],
                            content=str(msg.get("content", "")),
                            tool_calls=tool_calls,
                            tool_call_id=tool_call_id,
                            name=name,
                        )
                    )

        return messages

    def create_score(
        self,
        log_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Create a score/feedback for a Braintrust log entry.

        Args:
            log_id: The ID of the log entry to score
            name: The score name/type
            value: The score value
            comment: Optional comment explaining the score
            project_id: Project ID (overrides instance default)

        Returns:
            True if successful, False otherwise
        """
        project_id = project_id or self.project_id
        if not project_id:
            raise ValueError("project_id required")

        # Prepare feedback data - API expects "feedback" array
        feedback_item = {
            "id": log_id,
            "name": name,
            "value": value,
        }
        if comment:
            feedback_item["comment"] = comment

        feedback_data = {"feedback": [feedback_item]}

        # Post feedback to Braintrust
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            url = f"{self.api_url}/v1/project_logs/{project_id}/feedback"
            response = requests.post(url, headers=headers, json=feedback_data)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error creating Braintrust score: {e}")
            return False


def create_braintrust_adapter(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    project_id: Optional[str] = None,
) -> BraintrustAdapter:
    """Create a BraintrustAdapter instance."""
    return BraintrustAdapter(
        api_key=api_key,
        api_url=api_url,
        project_id=project_id,
    )


__all__ = ["scorer_to_reward_fn", "reward_fn_to_scorer", "BraintrustAdapter", "create_braintrust_adapter"]
