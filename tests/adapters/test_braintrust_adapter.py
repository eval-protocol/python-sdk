import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
import requests

from eval_protocol.adapters.braintrust import BraintrustAdapter
from eval_protocol.models import Message


class MockResponse:
    """Mock response object for requests.post"""

    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self.json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def mock_requests_post(monkeypatch):
    """Mock requests.post to return sample data"""

    def fake_post(url: str, headers=None, json=None):
        # Return a simplified response for basic tests
        return MockResponse(
            {
                "data": [
                    {
                        "id": "trace1",
                        "input": [{"role": "user", "content": "Hello"}],
                        "output": [{"message": {"role": "assistant", "content": "Hi there!"}}],
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", fake_post)
    return fake_post


def test_basic_btql_query_returns_evaluation_rows(mock_requests_post):
    """Test basic BTQL query execution and conversion to evaluation rows"""
    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")

    btql_query = "select: * from: project_logs('test_project') traces limit: 1"
    rows = adapter.get_evaluation_rows(btql_query)

    assert len(rows) == 1
    assert len(rows[0].messages) == 2
    assert rows[0].messages[0].role == "user"
    assert rows[0].messages[0].content == "Hello"
    assert rows[0].messages[1].role == "assistant"
    assert rows[0].messages[1].content == "Hi there!"


def test_trace_with_tool_calls_preserved(monkeypatch):
    """Test that tool calls are properly preserved in converted messages"""

    def mock_post(url: str, headers=None, json=None):
        return MockResponse(
            {
                "data": [
                    {
                        "id": "trace_with_tools",
                        "input": [{"role": "user", "content": "Get reservation details for 7KJ2PL"}],
                        "output": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_123",
                                            "type": "function",
                                            "function": {
                                                "name": "get_reservation_details",
                                                "arguments": '{"reservation_id": "7KJ2PL"}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", mock_post)

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    assert len(rows) == 1
    msgs = rows[0].messages

    # Find assistant message with tool calls
    assistant_msgs = [m for m in msgs if m.role == "assistant" and m.tool_calls]
    assert len(assistant_msgs) == 1

    assert assistant_msgs[0].tool_calls is not None
    tool_call = assistant_msgs[0].tool_calls[0]
    assert tool_call.id == "call_123"
    assert tool_call.function.name == "get_reservation_details"
    assert '{"reservation_id": "7KJ2PL"}' in tool_call.function.arguments


def test_trace_with_tool_response_messages(monkeypatch):
    """Test that tool response messages are properly handled"""

    def mock_post(url: str, headers=None, json=None):
        return MockResponse(
            {
                "data": [
                    {
                        "id": "trace_with_tool_response",
                        "input": [
                            {"role": "user", "content": "Check reservation"},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_456",
                                        "type": "function",
                                        "function": {
                                            "name": "get_reservation_details",
                                            "arguments": '{"reservation_id": "ABC123"}',
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call_456",
                                "content": '{"reservation_id": "ABC123", "status": "confirmed"}',
                            },
                        ],
                        "output": [
                            {"message": {"role": "assistant", "content": "Your reservation ABC123 is confirmed."}}
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", mock_post)

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    assert len(rows) == 1
    msgs = rows[0].messages

    # Should have user, assistant with tool_calls, tool response, and final assistant
    roles = [m.role for m in msgs]
    assert "user" in roles
    assert "tool" in roles
    assert roles.count("assistant") == 2  # One with tool_calls, one final response

    # Check tool message
    tool_msgs = [m for m in msgs if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_456"
    assert tool_msgs[0].content is not None
    assert "ABC123" in tool_msgs[0].content


def test_tools_extracted_from_metadata_variants(monkeypatch):
    """Test that tools are extracted from different metadata locations"""

    def mock_post_with_tools_in_metadata(url: str, headers=None, json=None):
        return MockResponse(
            {
                "data": [
                    {
                        "id": "trace_with_metadata_tools",
                        "input": [{"role": "user", "content": "Test"}],
                        "output": [{"message": {"role": "assistant", "content": "Response"}}],
                        "metadata": {
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {"name": "get_weather", "description": "Get weather info"},
                                }
                            ]
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", mock_post_with_tools_in_metadata)

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    assert len(rows) == 1
    assert rows[0].tools is not None
    assert len(rows[0].tools) == 1
    assert rows[0].tools[0]["function"]["name"] == "get_weather"


def test_tools_extracted_from_hidden_params(monkeypatch):
    """Test that tools are extracted from nested hidden_params location"""

    def mock_post_with_hidden_tools(url: str, headers=None, json=None):
        return MockResponse(
            {
                "data": [
                    {
                        "id": "trace_with_hidden_tools",
                        "input": [{"role": "user", "content": "Test"}],
                        "output": [{"message": {"role": "assistant", "content": "Response"}}],
                        "metadata": {
                            "hidden_params": {
                                "optional_params": {
                                    "tools": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "transfer_to_human_agents",
                                                "description": "Transfer to human",
                                            },
                                        }
                                    ]
                                }
                            }
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", mock_post_with_hidden_tools)

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    assert len(rows) == 1
    assert rows[0].tools is not None
    assert len(rows[0].tools) == 1
    assert rows[0].tools[0]["function"]["name"] == "transfer_to_human_agents"


def test_empty_btql_response_returns_empty_list(monkeypatch):
    """Test that empty BTQL response returns empty list"""

    def mock_empty_post(url: str, headers=None, json=None):
        return MockResponse({"data": []})

    monkeypatch.setattr("requests.post", mock_empty_post)

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    assert len(rows) == 0


def test_trace_without_meaningful_conversation_skipped(monkeypatch):
    """Test that traces without input or output are skipped"""

    def mock_post_incomplete_trace(url: str, headers=None, json=None):
        return MockResponse(
            {
                "data": [
                    {"id": "incomplete_trace", "input": None, "output": []},
                    {
                        "id": "valid_trace",
                        "input": [{"role": "user", "content": "Hello"}],
                        "output": [{"message": {"role": "assistant", "content": "Hi"}}],
                    },
                ]
            }
        )

    monkeypatch.setattr("requests.post", mock_post_incomplete_trace)

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    # Should only get the valid trace
    assert len(rows) == 1
    assert rows[0].input_metadata is not None
    assert rows[0].input_metadata.session_data is not None
    assert rows[0].input_metadata.session_data["braintrust_trace_id"] == "valid_trace"


def test_custom_converter_used_when_provided(monkeypatch):
    """Test that custom converter is used when provided"""

    def mock_post(url: str, headers=None, json=None):
        return MockResponse(
            {
                "data": [
                    {
                        "id": "custom_trace",
                        "input": [{"role": "user", "content": "Test"}],
                        "output": [{"message": {"role": "assistant", "content": "Response"}}],
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", mock_post)

    def custom_converter(trace: Dict[str, Any], include_tool_calls: bool):
        # Custom converter that adds a special message
        from eval_protocol.models import EvaluationRow, InputMetadata

        return EvaluationRow(
            messages=[Message(role="system", content="Custom converted message")],
            input_metadata=InputMetadata(session_data={"custom": True}),
        )

    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query", converter=custom_converter)

    assert len(rows) == 1
    assert rows[0].messages[0].role == "system"
    assert rows[0].messages[0].content == "Custom converted message"
    assert rows[0].input_metadata is not None
    assert rows[0].input_metadata.session_data is not None
    assert rows[0].input_metadata.session_data["custom"] is True


def test_api_authentication_error_handling(monkeypatch):
    """Test that API authentication errors are handled properly"""

    def mock_auth_error(url: str, headers=None, json=None):
        return MockResponse({}, status_code=401)

    monkeypatch.setattr("requests.post", mock_auth_error)

    adapter = BraintrustAdapter(api_key="invalid_key", project_id="test_project")

    with pytest.raises(requests.HTTPError):
        adapter.get_evaluation_rows("test query")


def test_session_data_includes_trace_id(mock_requests_post):
    """Test that session_data includes the Braintrust trace ID"""
    adapter = BraintrustAdapter(api_key="test_key", project_id="test_project")
    rows = adapter.get_evaluation_rows("test query")

    assert len(rows) == 1
    assert rows[0].input_metadata is not None
    assert rows[0].input_metadata.session_data is not None
    assert rows[0].input_metadata.session_data["braintrust_trace_id"] == "trace1"


def test_missing_required_env_vars(monkeypatch):
    """Test that missing required environment variables raise errors"""
    # Mock environment variables to be None
    monkeypatch.setenv("BRAINTRUST_API_KEY", "")
    monkeypatch.setenv("BRAINTRUST_PROJECT_ID", "")

    # Test missing API key
    with pytest.raises(ValueError, match="BRAINTRUST_API_KEY"):
        BraintrustAdapter(api_key=None, project_id="test_project")

    # Test missing project ID
    with pytest.raises(ValueError, match="BRAINTRUST_PROJECT_ID"):
        BraintrustAdapter(api_key="test_key", project_id=None)
