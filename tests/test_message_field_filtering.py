"""
Test to verify that message fields are properly filtered before sending to API.

This test verifies that unsupported fields like 'weight', 'control_plane_step',
and 'reasoning_content' are excluded from messages when preparing API requests.
"""

from eval_protocol.models import Message


def test_dump_model_excludes_unsupported_fields():
    """Test that dump_mdoel_for_chat_completion_request excludes unsupported fields."""
    # Create a message with all possible fields including unsupported ones
    message = Message(
        role="user",
        content="Hello",
        weight=0,
        control_plane_step={"step": 1},
        reasoning_content="Some reasoning",
        name="test_user",
    )

    # Get the filtered dictionary
    filtered = message.dump_mdoel_for_chat_completion_request()

    # Verify unsupported fields are excluded
    assert "weight" not in filtered, "weight field should be excluded"
    assert "control_plane_step" not in filtered, "control_plane_step field should be excluded"
    assert "reasoning_content" not in filtered, "reasoning_content field should be excluded"

    # Verify supported fields are included
    assert "role" in filtered, "role field should be included"
    assert "content" in filtered, "content field should be included"
    assert filtered["role"] == "user"
    assert filtered["content"] == "Hello"

    # Verify name is included (it's a supported field for tool calls)
    assert "name" in filtered
    assert filtered["name"] == "test_user"


def test_dump_model_with_only_supported_fields():
    """Test that supported fields are preserved."""
    message = Message(
        role="assistant",
        content="I can help you",
        tool_calls=None,
        tool_call_id=None,
    )

    filtered = message.dump_mdoel_for_chat_completion_request()

    # Should only contain supported fields
    assert filtered["role"] == "assistant"
    assert filtered["content"] == "I can help you"

    # Should not contain unsupported fields even if None
    assert "weight" not in filtered


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
