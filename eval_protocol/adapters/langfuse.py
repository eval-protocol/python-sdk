"""Langfuse adapter for Eval Protocol.

This adapter allows pulling data from Langfuse deployments and converting it
to EvaluationRow format for use in evaluation pipelines.
"""

from typing import Any, Dict, Iterator, List, Optional
from datetime import datetime
import logging

from eval_protocol.models import EvaluationRow, Message, InputMetadata, CompletionParams

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse

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

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        project_id: Optional[str] = None,
    ):
        """Initialize the Langfuse adapter.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL (default: https://cloud.langfuse.com)
            project_id: Optional project ID to filter traces
        """
        if not LANGFUSE_AVAILABLE:
            raise ImportError("Langfuse not installed. Install with: pip install 'eval-protocol[langfuse]'")

        self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self.project_id = project_id

    def get_evaluation_rows(
        self,
        limit: int = 100,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        include_tool_calls: bool = True,
    ) -> Iterator[EvaluationRow]:
        """Pull traces from Langfuse and convert to EvaluationRow format.

        Args:
            limit: Maximum number of rows to return
            tags: Filter by specific tags
            user_id: Filter by user ID
            session_id: Filter by session ID
            from_timestamp: Filter traces after this timestamp
            to_timestamp: Filter traces before this timestamp
            include_tool_calls: Whether to include tool calling traces

        Yields:
            EvaluationRow: Converted evaluation rows
        """
        # Get traces from Langfuse
        traces = self.client.get_traces(
            limit=limit,
            tags=tags,
            user_id=user_id,
            session_id=session_id,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )

        for trace in traces.data:
            try:
                eval_row = self._convert_trace_to_evaluation_row(trace, include_tool_calls)
                if eval_row:
                    yield eval_row
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to convert trace %s: %s", trace.id, e)
                continue

    def get_evaluation_rows_by_ids(
        self,
        trace_ids: List[str],
        include_tool_calls: bool = True,
    ) -> Iterator[EvaluationRow]:
        """Get specific traces by their IDs and convert to EvaluationRow format.

        Args:
            trace_ids: List of trace IDs to fetch
            include_tool_calls: Whether to include tool calling traces

        Yields:
            EvaluationRow: Converted evaluation rows
        """
        for trace_id in trace_ids:
            try:
                trace = self.client.get_trace(trace_id)
                eval_row = self._convert_trace_to_evaluation_row(trace, include_tool_calls)
                if eval_row:
                    yield eval_row
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to fetch/convert trace %s: %s", trace_id, e)
                continue

    def _convert_trace_to_evaluation_row(self, trace: Any, include_tool_calls: bool = True) -> Optional[EvaluationRow]:
        """Convert a Langfuse trace to EvaluationRow format.

        Args:
            trace: Langfuse trace object
            include_tool_calls: Whether to include tool calling information

        Returns:
            EvaluationRow or None if conversion fails
        """
        try:
            # Get observations (generations, spans) from the trace
            observations = self.client.get_observations(trace_id=trace.id).data

            # Convert observations to messages
            messages = self._extract_messages_from_observations(observations, include_tool_calls)

            if not messages:
                return None

            # Extract metadata
            input_metadata = self._create_input_metadata(trace, observations)

            # Extract ground truth if available (from trace metadata or tags)
            ground_truth = self._extract_ground_truth(trace)

            # Extract tools if available
            tools = self._extract_tools(observations) if include_tool_calls else None

            return EvaluationRow(
                messages=messages,
                tools=tools,
                input_metadata=input_metadata,
                ground_truth=ground_truth,
            )

        except (AttributeError, ValueError, KeyError) as e:
            logger.error("Error converting trace %s: %s", trace.id, e)
            return None

    def _extract_messages_from_observations(
        self, observations: List[Any], include_tool_calls: bool = True
    ) -> List[Message]:
        """Extract messages from Langfuse observations.

        Args:
            observations: List of Langfuse observation objects
            include_tool_calls: Whether to include tool calling information

        Returns:
            List of Message objects
        """
        messages = []

        # Sort observations by timestamp
        sorted_observations = sorted(observations, key=lambda x: x.start_time or datetime.min)

        for obs in sorted_observations:
            try:
                if hasattr(obs, "input") and obs.input:
                    # Handle different input formats
                    if isinstance(obs.input, dict):
                        if "messages" in obs.input:
                            # OpenAI-style messages format
                            for msg in obs.input["messages"]:
                                messages.append(self._dict_to_message(msg, include_tool_calls))
                        elif "role" in obs.input:
                            # Single message format
                            messages.append(self._dict_to_message(obs.input, include_tool_calls))
                        elif "prompt" in obs.input:
                            # Simple prompt format
                            messages.append(Message(role="user", content=str(obs.input["prompt"])))
                    elif isinstance(obs.input, str):
                        # Simple string input
                        messages.append(Message(role="user", content=obs.input))

                if hasattr(obs, "output") and obs.output:
                    # Handle output
                    if isinstance(obs.output, dict):
                        if "content" in obs.output:
                            messages.append(Message(role="assistant", content=str(obs.output["content"])))
                        elif "message" in obs.output:
                            msg_dict = obs.output["message"]
                            messages.append(self._dict_to_message(msg_dict, include_tool_calls))
                        else:
                            # Fallback: convert entire output to string
                            messages.append(Message(role="assistant", content=str(obs.output)))
                    elif isinstance(obs.output, str):
                        messages.append(Message(role="assistant", content=obs.output))

            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Error processing observation %s: %s", obs.id, e)
                continue

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

    def _create_input_metadata(self, trace: Any, observations: List[Any]) -> InputMetadata:
        """Create InputMetadata from trace and observations.

        Args:
            trace: Langfuse trace object
            observations: List of observation objects

        Returns:
            InputMetadata object
        """
        # Extract completion parameters from observations
        completion_params = CompletionParams()

        # Look for model parameters in observations
        for obs in observations:
            if hasattr(obs, "model") and obs.model:
                completion_params.model = obs.model
            if hasattr(obs, "model_parameters") and obs.model_parameters:
                params = obs.model_parameters
                if "temperature" in params:
                    completion_params.temperature = params["temperature"]
                if "max_tokens" in params:
                    completion_params.max_tokens = params["max_tokens"]
                if "top_p" in params:
                    completion_params.top_p = params["top_p"]
                break

        # Create dataset info from trace metadata
        dataset_info = {
            "trace_id": trace.id,
            "trace_name": getattr(trace, "name", None),
            "trace_tags": getattr(trace, "tags", []),
            "langfuse_project_id": self.project_id,
        }

        # Add trace metadata if available
        if hasattr(trace, "metadata") and trace.metadata:
            dataset_info["trace_metadata"] = trace.metadata

        # Create session data
        session_data = {
            "session_id": getattr(trace, "session_id", None),
            "user_id": getattr(trace, "user_id", None),
            "timestamp": getattr(trace, "timestamp", None),
            "langfuse_trace_url": (
                f"{self.client.host}/project/{self.project_id}/traces/{trace.id}" if self.project_id else None
            ),
        }

        return InputMetadata(
            row_id=trace.id,
            completion_params=completion_params,
            dataset_info=dataset_info,
            session_data=session_data,
        )

    def _extract_ground_truth(self, trace: Any) -> Optional[str]:
        """Extract ground truth from trace if available.

        Args:
            trace: Langfuse trace object

        Returns:
            Ground truth string or None
        """
        # Check trace metadata for ground truth
        if hasattr(trace, "metadata") and trace.metadata:
            if isinstance(trace.metadata, dict):
                return trace.metadata.get("ground_truth") or trace.metadata.get("expected_answer")

        # Check tags for ground truth indicators
        if hasattr(trace, "tags") and trace.tags:
            for tag in trace.tags:
                if tag.startswith("ground_truth:"):
                    return tag.replace("ground_truth:", "", 1)

        return None

    def _extract_tools(self, observations: List[Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract tool definitions from observations.

        Args:
            observations: List of observation objects

        Returns:
            List of tool definitions or None
        """
        tools = []

        for obs in observations:
            if hasattr(obs, "input") and obs.input and isinstance(obs.input, dict):
                if "tools" in obs.input:
                    tools.extend(obs.input["tools"])
                elif "functions" in obs.input:
                    # Convert functions to tools format
                    for func in obs.input["functions"]:
                        tools.append({"type": "function", "function": func})

        return tools if tools else None


def create_langfuse_adapter(
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com",
    project_id: Optional[str] = None,
) -> LangfuseAdapter:
    """Factory function to create a Langfuse adapter.

    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL
        project_id: Optional project ID

    Returns:
        LangfuseAdapter instance
    """
    return LangfuseAdapter(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        project_id=project_id,
    )
