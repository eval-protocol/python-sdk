from __future__ import annotations

import os
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from eval_protocol.human_id import generate_id
import json

from eval_protocol.models import Message


def _dbg_enabled() -> bool:
    return os.getenv("EP_DEBUG_SERIALIZATION", "0").strip() == "1"


def _dbg_print(*args):
    if _dbg_enabled():
        # Best-effort debug print without broad exception handling
        print(*args)


def serialize_lc_message_to_ep(msg: BaseMessage) -> Message:
    _dbg_print(
        "[EP-Ser] Input LC msg:",
        type(msg).__name__,
        {
            "has_additional_kwargs": isinstance(getattr(msg, "additional_kwargs", None), dict),
            "content_type": type(getattr(msg, "content", None)).__name__,
        },
    )

    if isinstance(msg, HumanMessage):
        ep_msg = Message(role="user", content=str(msg.content))
        _dbg_print("[EP-Ser] -> EP Message:", {"role": ep_msg.role, "len": len(ep_msg.content or "")})
        return ep_msg

    if isinstance(msg, AIMessage):
        # Extract visible content and hidden reasoning content if present
        content_text = ""
        reasoning_texts: List[str] = []

        if isinstance(msg.content, str):
            content_text = msg.content
        elif isinstance(msg.content, list):
            text_parts: List[str] = []
            for item in msg.content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif item_type in ("reasoning", "thinking", "thought"):
                        # Some providers return dedicated reasoning parts
                        maybe_text = item.get("text") or item.get("content")
                        if isinstance(maybe_text, str):
                            reasoning_texts.append(maybe_text)
                elif isinstance(item, str):
                    text_parts.append(item)
            content_text = "\n".join([t for t in text_parts if t])

        # Additional place providers may attach reasoning
        additional_kwargs = getattr(msg, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            rk = additional_kwargs.get("reasoning_content")
            if isinstance(rk, str) and rk:
                reasoning_texts.append(rk)

            # Fireworks and others sometimes nest under `reasoning` or `metadata`
            nested_reasoning = additional_kwargs.get("reasoning")
            if isinstance(nested_reasoning, dict):
                inner = nested_reasoning.get("content") or nested_reasoning.get("text")
                if isinstance(inner, str) and inner:
                    reasoning_texts.append(inner)

        # Capture tool calls and function_call if present on AIMessage
        def _normalize_tool_calls(raw_tcs):
            normalized = []
            for tc in raw_tcs or []:
                if isinstance(tc, dict) and "function" in tc:
                    # Assume already OpenAI style
                    fn = tc.get("function", {})
                    # Ensure arguments is a string
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        try:
                            args = json.dumps(args)
                        except Exception:
                            args = str(args)
                    normalized.append(
                        {
                            "id": tc.get("id") or generate_id(),
                            "type": tc.get("type") or "function",
                            "function": {"name": fn.get("name", ""), "arguments": args},
                        }
                    )
                elif isinstance(tc, dict) and ("name" in tc) and ("args" in tc or "arguments" in tc):
                    # LangChain tool schema → OpenAI function-call schema
                    name = tc.get("name", "")
                    args_val = tc.get("args", tc.get("arguments", {}))
                    if not isinstance(args_val, str):
                        try:
                            args_val = json.dumps(args_val)
                        except Exception:
                            args_val = str(args_val)
                    normalized.append(
                        {
                            "id": tc.get("id") or generate_id(),
                            "type": "function",
                            "function": {"name": name, "arguments": args_val},
                        }
                    )
                else:
                    # Best-effort: stringify unknown formats
                    normalized.append(
                        {
                            "id": generate_id(),
                            "type": "function",
                            "function": {
                                "name": str(tc.get("name", "tool")) if isinstance(tc, dict) else "tool",
                                "arguments": json.dumps(tc) if not isinstance(tc, str) else tc,
                            },
                        }
                    )
            return normalized if normalized else None

        extracted_tool_calls = None
        tc_attr = getattr(msg, "tool_calls", None)
        if isinstance(tc_attr, list):
            extracted_tool_calls = _normalize_tool_calls(tc_attr)

        if extracted_tool_calls is None and isinstance(additional_kwargs, dict):
            maybe_tc = additional_kwargs.get("tool_calls")
            if isinstance(maybe_tc, list):
                extracted_tool_calls = _normalize_tool_calls(maybe_tc)

        extracted_function_call = None
        fc_attr = getattr(msg, "function_call", None)
        if fc_attr:
            extracted_function_call = fc_attr
        if extracted_function_call is None and isinstance(additional_kwargs, dict):
            maybe_fc = additional_kwargs.get("function_call")
            if maybe_fc:
                extracted_function_call = maybe_fc

        ep_msg = Message(
            role="assistant",
            content=content_text,
            reasoning_content=("\n".join(reasoning_texts) if reasoning_texts else None),
            tool_calls=extracted_tool_calls,  # type: ignore[arg-type]
            function_call=extracted_function_call,  # type: ignore[arg-type]
        )
        _dbg_print(
            "[EP-Ser] -> EP Message:",
            {
                "role": ep_msg.role,
                "content_len": len(ep_msg.content or ""),
                "has_reasoning": bool(ep_msg.reasoning_content),
                "has_tool_calls": bool(ep_msg.tool_calls),
            },
        )
        return ep_msg

    if isinstance(msg, ToolMessage):
        tool_name = msg.name or "tool"
        status = msg.status or "success"
        content = str(msg.content)
        tool_call_id = getattr(msg, "tool_call_id", None)
        ep_msg = Message(
            role="tool",
            name=tool_name,
            tool_call_id=tool_call_id,
            content=f'<{tool_name} status="{status}">\n{content}\n</{tool_name}>',
        )
        _dbg_print(
            "[EP-Ser] -> EP Message:", {"role": ep_msg.role, "name": ep_msg.name, "has_id": bool(ep_msg.tool_call_id)}
        )
        return ep_msg

    ep_msg = Message(role=getattr(msg, "type", "assistant"), content=str(getattr(msg, "content", "")))
    _dbg_print("[EP-Ser] -> EP Message (fallback):", {"role": ep_msg.role, "len": len(ep_msg.content or "")})
    return ep_msg


def serialize_ep_messages_to_lc(messages: List[Message]) -> List[BaseMessage]:
    """Convert eval_protocol Message objects to LangChain BaseMessage list.

    - Flattens content parts into strings when content is a list
    - Maps EP roles to LC message classes
    """
    lc_messages: List[BaseMessage] = []
    for m in messages or []:
        content = m.content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                try:
                    text_parts.append(getattr(part, "text", ""))
                except AttributeError:
                    pass
            content = "\n".join([t for t in text_parts if t])
        if content is None:
            content = ""
        text = str(content)

        role = (m.role or "").lower()
        if role == "user":
            lc_messages.append(HumanMessage(content=text))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=text))
        elif role == "system":
            lc_messages.append(SystemMessage(content=text))
        else:
            lc_messages.append(HumanMessage(content=text))
    return lc_messages
