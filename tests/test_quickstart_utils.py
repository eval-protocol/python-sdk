"""Tests for quickstart utility functions."""

import pytest

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.utils.evaluation_row_utils import (
    multi_turn_assistant_to_ground_truth,
    serialize_message,
    assistant_to_ground_truth,
)


class TestSerializeMessage:
    """Tests for serialize_message function."""

    def test_simple_message(self):
        """Test serialization of a simple message."""
        message = Message(role="user", content="Hello, how are you?")
        result = serialize_message(message)
        assert result == "user: Hello, how are you?"

    def test_assistant_message(self):
        """Test serialization of an assistant message."""
        message = Message(role="assistant", content="I'm doing well, thank you!")
        result = serialize_message(message)
        assert result == "assistant: I'm doing well, thank you!"

    def test_message_with_tool_calls(self):
        """Test serialization of a message with tool calls."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "New York"}'},
        }
        message = Message(
            role="assistant",
            content="I'll check the weather for you.",
            tool_calls=[tool_call],  # pyright: ignore[reportArgumentType]
        )
        result = serialize_message(message)
        expected = 'assistant: I\'ll check the weather for you.\n[Tool Call: get_weather({"location": "New York"})]'
        assert result == expected

    def test_message_with_multiple_tool_calls(self):
        """Test serialization of a message with multiple tool calls."""
        tool_call1 = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
        }
        tool_call2 = {
            "id": "call_456",
            "type": "function",
            "function": {"name": "get_time", "arguments": '{"timezone": "EST"}'},
        }
        message = Message(
            role="assistant",
            content="Let me get both for you.",
            tool_calls=[tool_call1, tool_call2],  # pyright: ignore[reportArgumentType]
        )
        result = serialize_message(message)
        expected = (
            "assistant: Let me get both for you.\n"
            '[Tool Call: get_weather({"location": "NYC"})]\n'
            '[Tool Call: get_time({"timezone": "EST"})]'
        )
        assert result == expected

    def test_empty_content_message(self):
        """Test serialization of a message with empty content."""
        message = Message(role="assistant", content="")
        result = serialize_message(message)
        assert result == "assistant: "

    def test_none_content_message(self):
        """Test serialization of a message with None content."""
        message = Message(role="assistant", content=None)
        result = serialize_message(message)
        assert result == "assistant: None"


class TestMultiTurnAssistantToGroundTruth:
    """Tests for multi_turn_assistant_to_ground_truth function."""

    def test_single_turn_conversation(self):
        """Test that single-turn conversations are handled correctly."""
        messages = [
            Message(role="user", content="What's the weather like?"),
            Message(role="assistant", content="It's sunny today!"),
        ]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 1
        assert len(result[0].messages) == 1  # Only user message before assistant
        assert result[0].messages[0].role == "user"
        assert result[0].messages[0].content == "What's the weather like?"
        assert result[0].ground_truth == "assistant: It's sunny today!"

    def test_multi_turn_conversation(self):
        """Test that multi-turn conversations are split correctly."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well, thanks!"),
        ]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 2

        # First split: user -> assistant
        assert len(result[0].messages) == 1
        assert result[0].messages[0].content == "Hello"
        assert result[0].ground_truth == "assistant: Hi there!"

        # Second split: user -> assistant -> user -> assistant
        assert len(result[1].messages) == 3
        assert result[1].messages[0].content == "Hello"
        assert result[1].messages[1].content == "Hi there!"
        assert result[1].messages[2].content == "How are you?"
        assert result[1].ground_truth == "assistant: I'm doing well, thanks!"

    def test_conversation_with_system_message(self):
        """Test that system messages are preserved in splits."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well!"),
        ]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 2

        # First split should include system message
        assert len(result[0].messages) == 2
        assert result[0].messages[0].role == "system"
        assert result[0].messages[1].role == "user"

        # Second split should include system message and previous conversation
        assert len(result[1].messages) == 4
        assert result[1].messages[0].role == "system"

    def test_conversation_with_tool_calls(self):
        """Test that tool calls are preserved in ground truth."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
        }
        messages = [
            Message(role="user", content="What's the weather in NYC?"),
            Message(
                role="assistant",
                content="I'll check that for you.",
                tool_calls=[tool_call],  # pyright: ignore[reportArgumentType]
            ),
        ]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 1
        expected_ground_truth = 'assistant: I\'ll check that for you.\n[Tool Call: get_weather({"location": "NYC"})]'
        assert result[0].ground_truth == expected_ground_truth

    def test_multiple_rows_processing(self):
        """Test that multiple input rows are processed correctly."""
        row1 = EvaluationRow(
            messages=[Message(role="user", content="Hello"), Message(role="assistant", content="Hi!")]
        )
        row2 = EvaluationRow(
            messages=[Message(role="user", content="Goodbye"), Message(role="assistant", content="Bye!")]
        )

        result = multi_turn_assistant_to_ground_truth([row1, row2])

        assert len(result) == 2
        assert result[0].messages[0].content == "Hello"
        assert result[0].ground_truth == "assistant: Hi!"
        assert result[1].messages[0].content == "Goodbye"
        assert result[1].ground_truth == "assistant: Bye!"

    def test_no_assistant_messages(self):
        """Test that rows with no assistant messages return empty list."""
        messages = [Message(role="user", content="Hello"), Message(role="user", content="Anyone there?")]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 0

    def test_only_assistant_messages(self):
        """Test handling of rows with only assistant messages."""
        messages = [Message(role="assistant", content="Hello!"), Message(role="assistant", content="How can I help?")]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 2
        # First assistant message (no context)
        assert len(result[0].messages) == 0
        assert result[0].ground_truth == "assistant: Hello!"
        # Second assistant message (with first assistant as context)
        assert len(result[1].messages) == 1
        assert result[1].messages[0].content == "Hello!"
        assert result[1].ground_truth == "assistant: How can I help?"

    def test_duplicate_trace_filtering(self):
        """Test that duplicate traces are filtered out."""
        # Create two rows with the same conversation leading to different assistant responses
        messages1 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm good!"),
        ]
        messages2 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm great!"),  # Different response
        ]

        row1 = EvaluationRow(messages=messages1)
        row2 = EvaluationRow(messages=messages2)

        result = multi_turn_assistant_to_ground_truth([row1, row2])

        # Should only get 2 unique splits (not 4), because the context leading
        # to the second assistant message is the same in both rows
        assert len(result) == 2  # First "Hello" -> "Hi there!", then one unique context for second assistant

        # Verify the unique traces
        contexts = ["\n".join(serialize_message(m) for m in r.messages) for r in result]
        assert len(set(contexts)) == len(contexts)  # All contexts should be unique

    def test_tools_and_metadata_preservation(self):
        """Test that tools and input_metadata are preserved in split rows."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        input_metadata = InputMetadata(
            row_id="test_row", completion_params={"model": "gpt-4"}, session_data={"test": "data"}
        )

        messages = [Message(role="user", content="Hello"), Message(role="assistant", content="Hi!")]
        row = EvaluationRow(messages=messages, tools=tools, input_metadata=input_metadata)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 1
        assert result[0].tools == tools
        assert result[0].input_metadata == input_metadata

    def test_empty_input_list(self):
        """Test that empty input list returns empty result."""
        result = multi_turn_assistant_to_ground_truth([])
        assert len(result) == 0

    def test_complex_multi_turn_with_tool_responses(self):
        """Test complex conversation with tool calls and responses."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
        }

        messages = [
            Message(role="user", content="What's the weather in NYC?"),
            Message(
                role="assistant",
                content="I'll check that for you.",
                tool_calls=[tool_call],  # pyright: ignore[reportArgumentType]
            ),
            Message(role="tool", tool_call_id="call_123", content="Sunny, 75°F"),
            Message(role="assistant", content="It's sunny and 75°F in NYC!"),
            Message(role="user", content="Thanks!"),
            Message(role="assistant", content="You're welcome!"),
        ]
        row = EvaluationRow(messages=messages)

        result = multi_turn_assistant_to_ground_truth([row])

        assert len(result) == 3  # Three assistant messages

        # First assistant message with tool call
        assert len(result[0].messages) == 1  # Just user message
        assert "Tool Call: get_weather" in str(result[0].ground_truth or "")

        # Second assistant message after tool response
        assert len(result[1].messages) == 3  # user, assistant with tool call, tool response
        assert result[1].ground_truth == "assistant: It's sunny and 75°F in NYC!"

        # Third assistant message
        assert len(result[2].messages) == 5  # All previous messages + "Thanks!"
        assert result[2].ground_truth == "assistant: You're welcome!"


class TestAssistantToGroundTruth:
    """Tests for assistant_to_ground_truth function."""

    def test_removes_last_assistant_message(self):
        """Test that the last assistant message is removed and set as ground truth."""
        messages = [
            Message(role="user", content="What's the weather like?"),
            Message(role="assistant", content="It's sunny today!"),
        ]
        row = EvaluationRow(messages=messages)

        result = assistant_to_ground_truth([row])

        assert len(result) == 1
        assert len(result[0].messages) == 1  # Only user message remains
        assert result[0].messages[0].role == "user"
        assert result[0].messages[0].content == "What's the weather like?"
        assert result[0].ground_truth == "assistant: It's sunny today!"

    def test_multi_turn_with_last_assistant(self):
        """Test multi-turn conversation where last message is assistant."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well!"),
        ]
        row = EvaluationRow(messages=messages)

        result = assistant_to_ground_truth([row])

        assert len(result) == 1
        assert len(result[0].messages) == 3  # All except last assistant
        assert result[0].messages[-1].content == "How are you?"
        assert result[0].ground_truth == "assistant: I'm doing well!"

    def test_fails_when_last_message_not_assistant(self):
        """Test that function raises error when last message is not from assistant."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="Goodbye"),
        ]
        row = EvaluationRow(messages=messages)

        with pytest.raises(ValueError, match="Last message is not from assistant"):
            assistant_to_ground_truth([row])

    def test_preserves_metadata_and_tools(self):
        """Test that tools and metadata are preserved."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        tools = [{"type": "function", "function": {"name": "test"}}]
        input_metadata = InputMetadata(row_id="test_123", completion_params={})
        row = EvaluationRow(messages=messages, tools=tools, input_metadata=input_metadata)

        result = assistant_to_ground_truth([row])

        assert len(result) == 1
        assert result[0].tools == tools
        assert result[0].input_metadata == input_metadata
        assert result[0].ground_truth == "assistant: Hi there!"

    def test_multiple_rows(self):
        """Test processing multiple rows."""
        row1 = EvaluationRow(
            messages=[Message(role="user", content="Hello"), Message(role="assistant", content="Hi!")]
        )
        row2 = EvaluationRow(
            messages=[Message(role="user", content="Bye"), Message(role="assistant", content="Goodbye!")]
        )

        result = assistant_to_ground_truth([row1, row2])

        assert len(result) == 2
        assert result[0].ground_truth == "assistant: Hi!"
        assert result[1].ground_truth == "assistant: Goodbye!"

    def test_empty_input_list(self):
        """Test that empty input list returns empty result."""
        result = assistant_to_ground_truth([])
        assert len(result) == 0
