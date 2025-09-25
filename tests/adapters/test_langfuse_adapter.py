"""Tests for Langfuse adapter."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from eval_protocol.adapters.langfuse import (
    LangfuseAdapter,
    convert_trace_to_evaluation_row,
    extract_messages_from_trace,
)
from eval_protocol.models import EvaluationRow, InputMetadata, Message


class FakeLangfuseClient:
    """Mock Langfuse client for testing"""

    def __init__(self, traces_list_response, trace_details):
        self.traces_list_response = traces_list_response
        self.trace_details = trace_details

    @property
    def api(self):
        return FakeLangfuseAPI(self.traces_list_response, self.trace_details)

    def create_score(self, trace_id: str, name: str, value: float):
        """Mock score creation"""
        pass


class FakeLangfuseAPI:
    """Mock Langfuse API for testing"""

    def __init__(self, traces_list_response, trace_details):
        self.traces_list_response = traces_list_response
        self.trace_details = trace_details

    @property
    def trace(self):
        return FakeLangfuseTraceAPI(self.traces_list_response, self.trace_details)


class FakeLangfuseTraceAPI:
    """Mock Langfuse trace API for testing"""

    def __init__(self, traces_list_response, trace_details):
        self.traces_list_response = traces_list_response
        self.trace_details = trace_details

    def list(self, **kwargs):
        """Mock trace list method"""
        return self.traces_list_response

    def get(self, trace_id: str):
        """Mock trace get method"""
        return self.trace_details.get(trace_id, self.trace_details.get("default"))


def _create_mock_trace(
    trace_id: str, input_data: Any = None, output_data: Any = None, observations: Optional[List] = None
):
    """Helper to create mock trace objects"""
    # Ensure observations have metadata attribute
    obs_with_metadata = []
    for obs in observations or []:
        if hasattr(obs, "metadata"):
            obs_with_metadata.append(obs)
        else:
            # Add metadata to existing observation
            obs_dict = obs.__dict__ if hasattr(obs, "__dict__") else {}
            obs_dict["metadata"] = getattr(obs, "metadata", {})
            obs_with_metadata.append(Mock(**obs_dict))

    return Mock(
        id=trace_id, input=input_data, output=output_data, observations=obs_with_metadata, tags=[], metadata={}
    )


def _create_mock_traces_response(traces: List[Dict[str, Any]]):
    """Helper to create mock traces list response"""
    trace_objects = []
    for trace_data in traces:
        trace_objects.append(Mock(**trace_data))

    return Mock(data=trace_objects, meta=Mock(page=1, total_pages=1, total_items=len(trace_objects), limit=100))


@pytest.fixture
def mock_langfuse_client(monkeypatch):
    """Mock the Langfuse client"""

    def fake_get_client():
        traces_response = _create_mock_traces_response([{"id": "trace1", "name": "test_trace"}])
        trace_details = {
            "default": _create_mock_trace(
                "trace1",
                input_data={"messages": [{"role": "user", "content": "Hello"}]},
                output_data={"messages": [{"role": "assistant", "content": "Hi there!"}]},
            )
        }
        return FakeLangfuseClient(traces_response, trace_details)

    monkeypatch.setattr("eval_protocol.adapters.langfuse.get_client", fake_get_client)
    return fake_get_client


def test_basic_trace_conversion():
    """Test basic trace to evaluation row conversion"""
    trace = _create_mock_trace(
        "trace123",
        input_data={"messages": [{"role": "user", "content": "What's the weather?"}]},
        output_data={"messages": [{"role": "assistant", "content": "It's sunny!"}]},
    )
    result = convert_trace_to_evaluation_row(trace)

    assert result is not None
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "What's the weather?"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "It's sunny!"
    assert result.input_metadata is not None
    assert result.input_metadata.session_data is not None
    assert result.input_metadata.session_data["langfuse_trace_id"] == "trace123"


def test_trace_with_tool_calls():
    """Test trace conversion with tool calls"""
    trace = _create_mock_trace(
        "trace_tools",
        input_data={
            "messages": [{"role": "user", "content": "Get weather for NYC"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather info"}}],
        },
        output_data={
            "messages": [
                {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                        }
                    ],
                }
            ]
        },
    )

    result = convert_trace_to_evaluation_row(trace, include_tool_calls=True)

    assert result is not None
    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tools[0]["function"]["name"] == "get_weather"

    # Check that tool calls are preserved in messages
    assistant_msgs = [m for m in result.messages if m.role == "assistant" and m.tool_calls]
    assert len(assistant_msgs) == 1
    # Tool calls are converted to OpenAI format objects, not dicts
    assert assistant_msgs[0].tool_calls is not None
    tool_call = assistant_msgs[0].tool_calls[0]
    assert hasattr(tool_call, "function")
    assert tool_call.function.name == "get_weather"


def test_trace_conversion_with_span_name():
    """Test trace conversion with specific span name"""
    # Mock observations with spans and generations
    span_mock = Mock()
    span_mock.id = "span1"
    span_mock.name = "judge"
    span_mock.type = "SPAN"
    span_mock.metadata = {}
    span_mock.parent_observation_id = None

    gen_mock = Mock()
    gen_mock.id = "gen1"
    gen_mock.name = "generation"
    gen_mock.type = "GENERATION"
    gen_mock.parent_observation_id = "span1"
    gen_mock.input = {"messages": [{"role": "user", "content": "Judge this"}]}
    gen_mock.output = {"messages": [{"role": "assistant", "content": "Good response"}]}
    gen_mock.start_time = datetime.now()
    gen_mock.metadata = {}

    observations = [span_mock, gen_mock]

    trace = _create_mock_trace("trace_span", observations=observations)
    result = convert_trace_to_evaluation_row(trace, span_name="judge")

    assert result is not None
    assert len(result.messages) == 2
    assert result.messages[0].content == "Judge this"
    assert result.messages[1].content == "Good response"


def test_empty_trace_returns_none():
    """Test that empty traces return None"""
    trace = _create_mock_trace("empty_trace", input_data=None, output_data=None)

    result = convert_trace_to_evaluation_row(trace)

    assert result is None


def test_malformed_trace_returns_none():
    """Test that malformed traces are handled gracefully"""
    # Trace with missing required attributes
    trace = Mock(id="malformed")  # Missing input/output

    result = convert_trace_to_evaluation_row(trace)

    assert result is None


def test_langfuse_adapter_initialization(mock_langfuse_client):
    """Test LangfuseAdapter initialization"""
    adapter = LangfuseAdapter()
    assert adapter.client is not None


def test_langfuse_adapter_unavailable():
    """Test that ImportError is raised when Langfuse is not available"""
    import eval_protocol.adapters.langfuse as langfuse_module

    # Temporarily set LANGFUSE_AVAILABLE to False
    original_available = langfuse_module.LANGFUSE_AVAILABLE
    langfuse_module.LANGFUSE_AVAILABLE = False

    try:
        with pytest.raises(ImportError, match="Langfuse not installed"):
            LangfuseAdapter()
    finally:
        # Restore original value
        langfuse_module.LANGFUSE_AVAILABLE = original_available


def test_get_evaluation_rows_basic(mock_langfuse_client):
    """Test basic get_evaluation_rows functionality"""
    adapter = LangfuseAdapter()

    rows = adapter.get_evaluation_rows(limit=1)

    assert len(rows) == 1
    assert rows[0].messages[0].role == "user"
    assert rows[0].messages[0].content == "Hello"
    assert rows[0].input_metadata is not None
    assert rows[0].input_metadata.session_data is not None
    assert rows[0].input_metadata.session_data["langfuse_trace_id"] == "trace1"


def test_get_evaluation_rows_by_ids(mock_langfuse_client):
    """Test get_evaluation_rows_by_ids functionality"""
    adapter = LangfuseAdapter()

    rows = adapter.get_evaluation_rows_by_ids(["trace1"])

    assert len(rows) == 1
    assert rows[0].input_metadata is not None
    assert rows[0].input_metadata.session_data is not None
    assert rows[0].input_metadata.session_data["langfuse_trace_id"] == "trace1"


def test_get_evaluation_rows_by_ids_with_custom_converter(mock_langfuse_client):
    """Test get_evaluation_rows_by_ids with custom converter"""

    def custom_converter(trace, include_tool_calls: bool, span_name: Optional[str]):
        return EvaluationRow(
            messages=[Message(role="system", content="Custom converted message")],
            input_metadata=InputMetadata(session_data={"custom": True, "langfuse_trace_id": trace.id}),
        )

    adapter = LangfuseAdapter()
    rows = adapter.get_evaluation_rows_by_ids(["trace1"], converter=custom_converter)

    assert len(rows) == 1
    assert rows[0].messages[0].role == "system"
    assert rows[0].messages[0].content == "Custom converted message"
    assert rows[0].input_metadata is not None
    assert rows[0].input_metadata.session_data is not None
    assert rows[0].input_metadata.session_data["custom"] is True


def test_sampling_functionality(monkeypatch):
    """Test that sampling works correctly"""

    def fake_get_client():
        # Create multiple traces
        traces_response = _create_mock_traces_response([{"id": f"trace{i}", "name": f"trace_{i}"} for i in range(10)])
        trace_details = {
            f"trace{i}": _create_mock_trace(
                f"trace{i}",
                input_data={"messages": [{"role": "user", "content": f"Message {i}"}]},
                output_data={"messages": [{"role": "assistant", "content": f"Response {i}"}]},
            )
            for i in range(10)
        }
        trace_details["default"] = trace_details["trace0"]

        return FakeLangfuseClient(traces_response, trace_details)

    monkeypatch.setattr("eval_protocol.adapters.langfuse.get_client", fake_get_client)

    adapter = LangfuseAdapter()
    rows = adapter.get_evaluation_rows(limit=10, sample_size=3)

    # Should get exactly 3 rows due to sampling
    assert len(rows) == 3


def test_extract_messages_from_various_formats():
    """Test message extraction from different input formats"""
    # Test dict format with messages
    trace1 = _create_mock_trace(
        "trace1",
        input_data={"messages": [{"role": "user", "content": "Hello"}]},
        output_data={"messages": [{"role": "assistant", "content": "Hi"}]},
    )
    messages1 = extract_messages_from_trace(trace1)
    assert len(messages1) == 2
    assert messages1[0].role == "user"
    assert messages1[1].role == "assistant"

    # Test simple prompt format
    trace2 = _create_mock_trace(
        "trace2", input_data={"prompt": "What is AI?"}, output_data={"content": "AI is artificial intelligence"}
    )
    messages2 = extract_messages_from_trace(trace2)
    assert len(messages2) == 2
    assert messages2[0].role == "user"
    assert messages2[0].content == "What is AI?"
    assert messages2[1].role == "assistant"
    assert messages2[1].content == "AI is artificial intelligence"

    # Test list format
    trace3 = _create_mock_trace(
        "trace3",
        input_data=[{"role": "user", "content": "List format"}],
        output_data=[{"role": "assistant", "content": "Response"}],
    )
    messages3 = extract_messages_from_trace(trace3)
    assert len(messages3) == 2
    assert messages3[0].content == "List format"
    assert messages3[1].content == "Response"
