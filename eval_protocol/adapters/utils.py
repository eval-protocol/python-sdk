"""Common utilities for adapter implementations.

This module contains shared functions and utilities used across different
adapter implementations to avoid code duplication.
"""

import logging
import time
from typing import Any, Dict, List

from eval_protocol.models import Message

logger = logging.getLogger(__name__)


def extract_messages_from_data(data, include_tool_calls: bool) -> List[Message]:
    """Extract messages from data (works for both input and output).

    This is a common function used by multiple adapters to parse message data
    from various formats (dict, list, string) into standardized Message objects.

    Args:
        data: Data from trace/log (input or output) - can be dict, list, or string
        include_tool_calls: Whether to include tool calling information

    Returns:
        List of Message objects
    """
    messages = []

    if isinstance(data, dict):
        if "messages" in data:
            # OpenAI-style messages format
            for msg in data["messages"]:
                messages.append(dict_to_message(msg, include_tool_calls))
        elif "role" in data:
            # Single message format
            messages.append(dict_to_message(data, include_tool_calls))
        elif "prompt" in data:
            # Simple prompt format
            messages.append(Message(role="user", content=str(data["prompt"])))
        elif "content" in data:
            # Simple content format
            messages.append(Message(role="assistant", content=str(data["content"])))
        else:
            # Fallback: treat as single message
            messages.append(dict_to_message(data, include_tool_calls))
    elif isinstance(data, list):
        # Direct list of message dicts
        for msg in data:
            if isinstance(msg, dict):
                messages.append(dict_to_message(msg, include_tool_calls))
    elif isinstance(data, str):
        # Simple string - role depends on context, default to user
        messages.append(Message(role="user", content=data))

    return messages


def dict_to_message(msg_dict: Dict[str, Any], include_tool_calls: bool = True) -> Message:
    """Convert a dictionary to a Message object.

    This is a common function used by multiple adapters to convert dictionary
    representations of messages into standardized Message objects.

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
