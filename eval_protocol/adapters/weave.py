"""Weave (Weights & Biases) adapter for Eval Protocol.

This adapter fetches recent root traces from Weave Trace API and converts them
to `EvaluationRow` format for use in evaluation pipelines. It is intentionally
minimal and depends only on requests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import requests

from eval_protocol.models import EvaluationRow, InputMetadata, Message, ExecutionMetadata
from .base import BaseAdapter


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


def _convert_trace_to_evaluation_row(
    trace: Dict[str, Any], include_tool_calls: bool = True
) -> Optional[EvaluationRow]:
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
        messages=messages, tools=tools, input_metadata=input_metadata, execution_metadata=execution_metadata
    )


class WeaveAdapter(BaseAdapter):
    """Adapter to pull data from Weave Trace API and convert to EvaluationRow format."""

    def __init__(
        self, base_url: Optional[str] = None, api_token: Optional[str] = None, project_id: Optional[str] = None
    ):
        self.base_url = base_url or os.getenv("WEAVE_TRACE_BASE_URL", "https://trace.wandb.ai")
        self.api_token = api_token or os.getenv("WANDB_API_KEY")
        # project_id is in form "<entity>/<project>"
        self.project_id = project_id or (f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}")
        if not self.api_token or not self.project_id or "/" not in self.project_id:
            raise ValueError("Missing Weave credentials or project (WANDB_API_KEY and WANDB_ENTITY/WANDB_PROJECT)")

    def _fetch_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/calls/stream_query"
        payload = {
            "project_id": self.project_id,
            "filter": {"trace_roots_only": True},
            "limit": limit,
            "offset": 0,
            "sort_by": [{"field": "started_at", "direction": "desc"}],
            "include_feedback": False,
        }
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json() or {}
        return body.get("data", [])

    def get_evaluation_rows(self, *args, **kwargs) -> List[EvaluationRow]:
        limit = kwargs.get("limit", 100)
        include_tool_calls = kwargs.get("include_tool_calls", True)
        traces = self._fetch_traces(limit=limit)
        rows: List[EvaluationRow] = []
        for tr in traces:
            row = _convert_trace_to_evaluation_row(tr, include_tool_calls=include_tool_calls)
            if row:
                rows.append(row)
        return rows
