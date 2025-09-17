"""LangSmith adapter for Eval Protocol.

This adapter pulls runs from LangSmith and converts them to EvaluationRow format,
mirroring the behavior of the Langfuse adapter.

It supports extracting chat messages from inputs/outputs, and optionally includes
tool calls and tool messages where present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Iterable, cast

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from .base import BaseAdapter

logger = logging.getLogger(__name__)

try:
    from langsmith import Client  # type: ignore

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None  # type: ignore[misc]


class LangSmithAdapter(BaseAdapter):
    """Adapter to pull data from LangSmith and convert to EvaluationRow format.

    By default, fetches root runs from a project and maps inputs/outputs into
    `Message` objects. It supports a variety of input/output shapes commonly
    emitted by LangChain/LangGraph integrations, including:
    - inputs: { messages: [...] } | { prompt } | { user_input } | { input } | str | list[dict]
    - outputs: { messages: [...] } | { content } | { result } | { answer } | { output } | str | list[dict]
    """

    def __init__(self, client: Optional[Any] = None) -> None:
        if not LANGSMITH_AVAILABLE:
            raise ImportError("LangSmith not installed. Install with: pip install 'eval-protocol[langsmith]'")
        if client is not None:
            self.client = client
        else:
            assert Client is not None
            self.client = cast(Any, Client)()

    def get_evaluation_rows(
        self,
        *,
        project_name: str,
        limit: int = 50,
        include_tool_calls: bool = True,
        # Pass-through filters to list_runs to match LangSmith Client API
        run_id: Optional[str] = None,
        ids: Optional[List[str]] = None,
        run_type: Optional[str] = None,
        execution_order: Optional[int] = None,
        parent_run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        trace_ids: Optional[List[str]] = None,
        reference_example_id: Optional[str] = None,
        session_name: Optional[str] = None,
        error: Optional[bool] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        filter_expr: Optional[str] = None,  # server-side filter DSL
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feedback_keys: Optional[List[str]] = None,
        feedback_source: Optional[str] = None,
        tree_id: Optional[str] = None,
        # ordering/pagination
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        # selection
        select: Optional[List[str]] = None,
        **list_runs_kwargs: Any,
    ) -> List[EvaluationRow]:
        """Pull runs from LangSmith and convert to EvaluationRow format.

        Args:
            project_name: LangSmith project to read runs from
            limit: Maximum number of rows to return
            include_tool_calls: Whether to include tool calling information when present
        """
        rows: List[EvaluationRow] = []

        # Fetch runs with pass-through filters. Prefer root runs by default.
        params: Dict[str, Any] = {"project_name": project_name, "limit": limit}
        # Only include non-None params
        if run_type is None:
            params["is_root"] = True
        for key, value in [
            ("id", run_id),
            ("ids", ids),
            ("run_type", run_type),
            ("execution_order", execution_order),
            ("parent_run_id", parent_run_id),
            ("trace_id", trace_id),
            ("trace_ids", trace_ids),
            ("reference_example_id", reference_example_id),
            ("session_name", session_name),
            ("error", error),
            ("start_time", start_time),
            ("end_time", end_time),
            ("filter", filter_expr),
            ("tags", tags),
            ("metadata", metadata),
            ("feedback_keys", feedback_keys),
            ("feedback_source", feedback_source),
            ("tree_id", tree_id),
            ("offset", offset),
            ("order_by", order_by),
        ]:
            if value is not None:
                params[key] = value
        params["select"] = select or ["id", "inputs", "outputs", "trace_id"]

        # Merge any additional kwargs last to allow explicit overrides
        if list_runs_kwargs:
            for k, v in list_runs_kwargs.items():
                if v is not None:
                    params[k] = v

        runs_iter: Iterable[Any] = self.client.list_runs(**params)

        runs = list(runs_iter)
        if not runs:
            logger.warning("No LangSmith runs found for project '%s' with current filters", project_name)
            return []

        # Group by trace_id and pick the last run in each trace (assume iterator yields chronological)
        trace_to_last_run: Dict[str, Any] = {}
        for r in runs:
            t_id = str(getattr(r, "trace_id", "")) or str(getattr(r, "id", ""))
            trace_to_last_run[t_id] = r

        for r in trace_to_last_run.values():
            try:
                inp = getattr(r, "inputs", None)
                out = getattr(r, "outputs", None)

                ep_messages: List[Message] = []
                # Prefer canonical conversation from outputs.messages if present to avoid duplicates
                if isinstance(out, dict) and isinstance(out.get("messages"), list):
                    ep_messages.extend(
                        self._extract_messages_from_payload(
                            {"messages": out["messages"]}, include_tool_calls, is_output=True
                        )
                    )
                else:
                    # Inputs → user messages
                    ep_messages.extend(self._extract_messages_from_payload(inp, include_tool_calls))
                    # Outputs → assistant (and possible tool messages)
                    ep_messages.extend(self._extract_messages_from_payload(out, include_tool_calls, is_output=True))

                # Deduplicate consecutive identical user messages (common echo pattern)
                def _canon(text: Any) -> str:
                    # Best-effort canonicalization; avoid broad exception handling warnings by handling types
                    text_str = str(text) if text is not None else ""
                    return " ".join(text_str.strip().lower().split())

                deduped: List[Message] = []
                for m in ep_messages:
                    if deduped and m.role == "user" and deduped[-1].role == "user":
                        if _canon(m.content) == _canon(deduped[-1].content):
                            continue
                    deduped.append(m)
                ep_messages = deduped

                if not ep_messages:
                    continue

                tools = None
                if include_tool_calls and isinstance(inp, dict):
                    # Try to extract tool schema if present in inputs
                    if "tools" in inp:
                        tools = inp["tools"]

                rows.append(
                    EvaluationRow(
                        messages=ep_messages,
                        tools=tools,
                        input_metadata=InputMetadata(
                            session_data={
                                "langsmith_run_id": str(getattr(r, "id", "")),
                                "langsmith_trace_id": str(getattr(r, "trace_id", "")),
                                "langsmith_project": project_name,
                            }
                        ),
                    )
                )
            except (AttributeError, ValueError, KeyError, TypeError) as e:
                logger.warning("Failed to convert run %s: %s", getattr(r, "id", ""), e)
                continue

        return rows

    def get_evaluation_rows_by_ids(
        self,
        *,
        run_ids: Optional[List[str]] = None,
        trace_ids: Optional[List[str]] = None,
        include_tool_calls: bool = True,
        project_name: Optional[str] = None,
    ) -> List[EvaluationRow]:
        """Fetch specific runs or traces and convert to EvaluationRow.

        If both run_ids and trace_ids are provided, both sets are fetched.
        """
        results: List[EvaluationRow] = []

        fetched_runs: List[Any] = []
        try:
            if run_ids:
                fetched_runs.extend(
                    list(self.client.list_runs(ids=run_ids, select=["id", "inputs", "outputs", "trace_id"]))
                )
            if trace_ids:
                fetched_runs.extend(
                    list(self.client.list_runs(trace_ids=trace_ids, select=["id", "inputs", "outputs", "trace_id"]))
                )
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to fetch runs by ids: %s", e)
            return []

        if not fetched_runs:
            logger.warning("No LangSmith runs found for provided ids")
            return []

        # Prefer the last run per trace id
        trace_to_last_run: Dict[str, Any] = {}
        for r in fetched_runs:
            t_id = str(getattr(r, "trace_id", "")) or str(getattr(r, "id", ""))
            trace_to_last_run[t_id] = r

        for r in trace_to_last_run.values():
            try:
                inp = getattr(r, "inputs", None)
                out = getattr(r, "outputs", None)

                ep_messages: List[Message] = []
                if isinstance(out, dict) and isinstance(out.get("messages"), list):
                    ep_messages.extend(
                        self._extract_messages_from_payload(
                            {"messages": out["messages"]}, include_tool_calls, is_output=True
                        )
                    )
                else:
                    ep_messages.extend(self._extract_messages_from_payload(inp, include_tool_calls))
                    ep_messages.extend(self._extract_messages_from_payload(out, include_tool_calls, is_output=True))

                def _canon(text: Any) -> str:
                    text_str = str(text) if text is not None else ""
                    return " ".join(text_str.strip().lower().split())

                deduped: List[Message] = []
                for m in ep_messages:
                    if deduped and m.role == "user" and deduped[-1].role == "user":
                        if _canon(m.content) == _canon(deduped[-1].content):
                            continue
                    deduped.append(m)
                ep_messages = deduped

                if not ep_messages:
                    continue

                tools = None
                if include_tool_calls and isinstance(inp, dict) and "tools" in inp:
                    tools = inp["tools"]

                results.append(
                    EvaluationRow(
                        messages=ep_messages,
                        tools=tools,
                        input_metadata=InputMetadata(
                            session_data={
                                "langsmith_run_id": str(getattr(r, "id", "")),
                                "langsmith_trace_id": str(getattr(r, "trace_id", "")),
                                "langsmith_project": project_name or "",
                            }
                        ),
                    )
                )
            except (AttributeError, ValueError, KeyError, TypeError) as e:
                logger.warning("Failed to convert run %s: %s", getattr(r, "id", ""), e)
                continue

        return results

    def _extract_messages_from_payload(
        self, payload: Any, include_tool_calls: bool, *, is_output: bool = False
    ) -> List[Message]:
        messages: List[Message] = []

        def _dict_to_message(msg_dict: Dict[str, Any]) -> Message:
            # Role
            role = msg_dict.get("role")
            if role is None:
                # Map LangChain types to roles if available
                msg_type = msg_dict.get("type")
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
                else:
                    role = "assistant" if is_output else "user"

            content = msg_dict.get("content")
            # LangChain content parts
            if isinstance(content, list):
                text = " ".join([part.get("text", "") for part in content if isinstance(part, dict)])
                content = text or str(content)

            name = msg_dict.get("name")

            tool_calls = None
            tool_call_id = None
            function_call = None
            if include_tool_calls:
                if "tool_calls" in msg_dict and isinstance(msg_dict["tool_calls"], list):
                    try:
                        from openai.types.chat.chat_completion_message_tool_call import (
                            ChatCompletionMessageToolCall,
                            Function as ChatToolFunction,
                        )

                        typed_calls: List[ChatCompletionMessageToolCall] = []
                        for tc in msg_dict["tool_calls"]:
                            # Extract id/type/function fields from dicts or provider-native objects
                            if isinstance(tc, dict):
                                tc_id = tc.get("id", None)
                                fn = tc.get("function", {}) or {}
                                fn_name = fn.get("name", None)
                                fn_args = fn.get("arguments", None)
                            else:
                                tc_id = getattr(tc, "id", None)
                                f = getattr(tc, "function", None)
                                fn_name = getattr(f, "name", None) if f is not None else None
                                fn_args = getattr(f, "arguments", None) if f is not None else None

                            # Build typed function object (arguments must be a string per OpenAI type)
                            fn_obj = ChatToolFunction(
                                name=str(fn_name) if fn_name is not None else "",
                                arguments=str(fn_args) if fn_args is not None else "",
                            )
                            typed_calls.append(
                                ChatCompletionMessageToolCall(
                                    id=str(tc_id) if tc_id is not None else "",
                                    type="function",
                                    function=fn_obj,
                                )
                            )
                        tool_calls = typed_calls
                    except (ImportError, AttributeError, TypeError, ValueError):
                        # If OpenAI types unavailable, leave None to satisfy type checker
                        tool_calls = None
                if "tool_call_id" in msg_dict:
                    tool_call_id = msg_dict.get("tool_call_id")
                if "function_call" in msg_dict:
                    function_call = msg_dict.get("function_call")

            return Message(
                role=str(role),
                content=str(content) if content is not None else "",
                name=name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
                function_call=function_call,
            )

        if isinstance(payload, dict):
            # Common patterns
            if isinstance(payload.get("messages"), list):
                for m in payload["messages"]:
                    if isinstance(m, dict):
                        messages.append(_dict_to_message(m))
                    else:
                        messages.append(Message(role="assistant" if is_output else "user", content=str(m)))
            elif "prompt" in payload and isinstance(payload["prompt"], str):
                messages.append(Message(role="user" if not is_output else "assistant", content=str(payload["prompt"])))
            elif "user_input" in payload and isinstance(payload["user_input"], str):
                messages.append(
                    Message(role="user" if not is_output else "assistant", content=str(payload["user_input"]))
                )
            elif "input" in payload and isinstance(payload["input"], str):
                messages.append(Message(role="user" if not is_output else "assistant", content=str(payload["input"])))
            elif "content" in payload and isinstance(payload["content"], str):
                messages.append(Message(role="assistant", content=str(payload["content"])))
            elif "result" in payload and isinstance(payload["result"], str):
                messages.append(Message(role="assistant", content=str(payload["result"])))
            elif "answer" in payload and isinstance(payload["answer"], str):
                messages.append(Message(role="assistant", content=str(payload["answer"])))
            elif "output" in payload and isinstance(payload["output"], str):
                messages.append(Message(role="assistant", content=str(payload["output"])))
            else:
                # Fallback: stringify
                messages.append(Message(role="assistant" if is_output else "user", content=str(payload)))
        elif isinstance(payload, list):
            for m in payload:
                if isinstance(m, dict):
                    messages.append(_dict_to_message(m))
                else:
                    messages.append(Message(role="assistant" if is_output else "user", content=str(m)))
        elif isinstance(payload, str):
            messages.append(Message(role="assistant" if is_output else "user", content=payload))

        return messages


def create_langsmith_adapter() -> LangSmithAdapter:
    return LangSmithAdapter()
