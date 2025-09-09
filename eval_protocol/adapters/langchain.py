from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from eval_protocol.models import Message


def _dbg_enabled() -> bool:
    return os.getenv("EP_DEBUG_SERIALIZATION", "0").strip() == "1"


def _dbg_print(*args):
    if _dbg_enabled():
        try:
            print(*args)
        except Exception:
            pass


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
        content = ""
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            parts: List[str] = []
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)

        ep_msg = Message(role="assistant", content=content)
        _dbg_print(
            "[EP-Ser] -> EP Message:",
            {
                "role": ep_msg.role,
                "content_len": len(ep_msg.content or ""),
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
            from langchain_core.messages import SystemMessage  # local import to avoid unused import

            lc_messages.append(SystemMessage(content=text))
        else:
            lc_messages.append(HumanMessage(content=text))
    return lc_messages
