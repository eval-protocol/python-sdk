"""LangSmith adapter for Eval Protocol.

This adapter pulls runs from LangSmith and converts them to EvaluationRow format,
mirroring the behavior of the Langfuse adapter.

It supports extracting chat messages from inputs/outputs, and optionally includes
tool calls and tool messages where present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluationRow, InputMetadata, Message

logger = logging.getLogger(__name__)

try:
    from langsmith import Client  # type: ignore

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


class LangSmithAdapter:
    """Adapter to pull data from LangSmith and convert to EvaluationRow format.

    By default, fetches root runs from a project and maps inputs/outputs into
    `Message` objects. It supports a variety of input/output shapes commonly
    emitted by LangChain/LangGraph integrations, including:
    - inputs: { messages: [...] } | { prompt } | { user_input } | { input } | str | list[dict]
    - outputs: { messages: [...] } | { content } | { result } | { answer } | { output } | str | list[dict]
    """

    def __init__(self, client: Optional[Client] = None) -> None:
        if not LANGSMITH_AVAILABLE:
            raise ImportError("LangSmith not installed. Install with: pip install langsmith")
        self.client = client or Client()

    def get_evaluation_rows(
        self,
        *,
        project_name: str,
        limit: int = 50,
        include_tool_calls: bool = True,
    ) -> List[EvaluationRow]:
        """Pull runs from LangSmith and convert to EvaluationRow format.

        Args:
            project_name: LangSmith project to read runs from
            limit: Maximum number of rows to return
            include_tool_calls: Whether to include tool calling information when present
        """
        rows: List[EvaluationRow] = []

        # Prefer root runs; they usually contain messages in inputs/outputs when tracing app-level flows
        runs = list(
            self.client.list_runs(
                project_name=project_name,
                is_root=True,
                limit=limit,
                select=["id", "inputs", "outputs"],
            )
        )

        for r in runs:
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
                    try:
                        return " ".join(str(text or "").strip().lower().split())
                    except Exception:
                        return str(text or "")

                deduped: List[Message] = []
                for m in ep_messages:
                    if deduped and m.role == "user" and deduped[-1].role == "user":
                        if _canon(m.content) == _canon(deduped[-1].content):
                            continue
                    deduped.append(m)
                ep_messages = deduped

                if not ep_messages:
                    continue

                rows.append(
                    EvaluationRow(
                        messages=ep_messages,
                        input_metadata=InputMetadata(
                            session_data={
                                "langsmith_run_id": str(getattr(r, "id", "")),
                                "langsmith_project": project_name,
                            }
                        ),
                    )
                )
            except Exception as e:
                logger.warning("Failed to convert run %s: %s", getattr(r, "id", ""), e)
                continue

        return rows

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
                                tc_type = tc.get("type", "function") or "function"
                                fn = tc.get("function", {}) or {}
                                fn_name = fn.get("name", None)
                                fn_args = fn.get("arguments", None)
                            else:
                                tc_id = getattr(tc, "id", None)
                                tc_type = getattr(tc, "type", None) or "function"
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
                    except Exception:
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
