from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluationRow, InputMetadata, Message, ExecutionMetadata


def _extract_messages_from_trace(trace: Dict[str, Any], include_tool_calls: bool = True) -> List[Message]:
    messages: List[Message] = []

    # Prefer explicit output messages if provided
    output = trace.get("output") or {}
    out_msgs = output.get("messages")
    if isinstance(out_msgs, list):
        for m in out_msgs:
            messages.append(
                Message(
                    role=m.get("role"),
                    content=m.get("content"),
                    tool_calls=m.get("tool_calls") if include_tool_calls else None,
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                )
            )

    # If no explicit output messages, fall back to final bubble from choices
    if not messages:
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            msg = (choices[0] or {}).get("message", {})
            if msg:
                messages.append(Message(role=msg.get("role"), content=msg.get("content")))

    # Prepend input messages if present and not already contained
    inputs = trace.get("inputs") or {}
    in_msgs = inputs.get("messages")
    if isinstance(in_msgs, list):
        prefixed = [Message(role=m.get("role"), content=m.get("content")) for m in in_msgs]
        messages = prefixed + messages

    return messages


def convert_trace_to_evaluation_row(trace: Dict[str, Any], include_tool_calls: bool = True) -> Optional[EvaluationRow]:
    messages = _extract_messages_from_trace(trace, include_tool_calls=include_tool_calls)
    if not messages:
        return None

    # Provider-native IDs for UI joinability
    session_data = {
        "weave_trace_id": trace.get("id"),
        "weave_project_id": trace.get("project_id"),
    }

    # Optional EP identifiers (if present in provider payload)
    meta_in = (trace.get("inputs") or {}).get("metadata") or {}
    meta_out = (trace.get("output") or {}).get("metadata") or {}
    metadata = {**meta_in, **meta_out}

    input_metadata = InputMetadata(row_id=metadata.get("row_id"), session_data=session_data)

    # Preserve default factory behavior by only setting provided fields
    exec_kwargs: Dict[str, Any] = {}
    for k in ("invocation_id", "experiment_id", "rollout_id", "run_id"):
        if metadata.get(k) is not None:
            exec_kwargs[k] = metadata[k]
    execution_metadata = ExecutionMetadata(**exec_kwargs)

    # Capture tools if provider exposes them (prefer inputs)
    tools = None
    inputs = trace.get("inputs") or {}
    if include_tool_calls and isinstance(inputs, dict) and "tools" in inputs:
        tools = inputs.get("tools")

    return EvaluationRow(
        messages=messages,
        tools=tools,
        input_metadata=input_metadata,
        execution_metadata=execution_metadata,
    )
