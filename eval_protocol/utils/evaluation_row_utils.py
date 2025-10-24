"""
Utility functions for processing and transforming EvaluationRow objects.

This module contains functions that work with EvaluationRow objects for various
preprocessing, filtering, and transformation tasks commonly used across the
evaluation pipeline.
"""

from typing import List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.models import InputMetadata


def serialize_message(msg: Message) -> str:
    """
    Convert a Message object to a string representation.

    Args:
        msg: Message object to serialize

    Returns:
        String representation of the message including role, content, and tool calls
    """
    parts = [f"{msg.role}: {msg.content}"]

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            parts.append(f"[Tool Call: {tool_name}({tool_args})]")

    return "\n".join(parts)


def filter_longest_conversation(data: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Filter out the longest conversation from a list of evaluation rows that share the same rollout_id.

    Args:
        data: List of EvaluationRow objects that share the same rollout_id

    Returns:
        List containing only the EvaluationRow with the most messages (longest conversation)
    """
    if not data:
        return data

    if len(data) == 1:
        return data

    # Find the row with the most messages (longest conversation)
    longest_row = max(data, key=lambda row: len(row.messages))

    return [longest_row]


def multi_turn_assistant_to_ground_truth(data: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Split multi-turn conversations into rows, with each assistant message as ground truth.

    Args:
        data: List of EvaluationRow objects

    Returns:
        List of expanded EvaluationRow objects, one for each assistant message
    """
    expanded_rows = []
    seen_traces: set[str] = set()

    for row in data:
        messages = row.messages
        tools = row.tools
        input_metadata = row.input_metadata

        assistant_positions = []
        for i, message in enumerate(messages):
            if message.role == "assistant":
                assistant_positions.append(i)

        # Create separate evaluation rows on each assistant message (where the comparison model will respond)
        for pos in assistant_positions:
            messages_before_assistant = messages[:pos]
            assistant_message = messages[pos]

            # In this case, we trace every request, so we need to filter out duplicates
            curr_trace = "\n".join(serialize_message(m) for m in messages_before_assistant)
            if curr_trace in seen_traces:
                continue
            seen_traces.add(curr_trace)

            ground_truth_message = serialize_message(assistant_message)

            expanded_rows.append(
                EvaluationRow(
                    messages=messages_before_assistant,
                    tools=tools,
                    input_metadata=input_metadata,
                    ground_truth=ground_truth_message,
                )
            )

    return expanded_rows


def assistant_to_ground_truth(data: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Extract the last assistant message as ground truth and remove it from the conversation.

    Args:
        data: List of EvaluationRow objects

    Returns:
        List of EvaluationRow objects with last assistant message moved to ground_truth
    """
    processed_rows = []

    for row in data:
        messages = row.messages.copy()  # Don't modify original

        if messages[-1].role == "assistant":
            assistant_message = messages[-1]
            messages = messages[:-1]
            ground_truth_message = serialize_message(assistant_message)
        else:
            raise ValueError("Last message is not from assistant")

        processed_rows.append(
            EvaluationRow(
                messages=messages,
                tools=row.tools,
                input_metadata=row.input_metadata,
                ground_truth=ground_truth_message,
            )
        )

    return processed_rows


def create_rows_from_indices(count: int, **metadata) -> List[EvaluationRow]:
    """Create evaluation rows with sequential row_ids.
    Useful for remote processors where the server determines content based on row_id.
    Args:
        count: Number of rows to create
        **metadata: Additional metadata to include in each row
    Returns:
        List of EvaluationRows with row_id set to "0", "1", "2", ...
    """
    rows = []
    for idx in range(count):
        row_metadata = {**metadata, "row_id": str(idx)}
        rows.append(
            EvaluationRow(
                messages=[],
                input_metadata=InputMetadata(**row_metadata),
            )
        )
    return rows
