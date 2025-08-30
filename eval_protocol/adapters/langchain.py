from __future__ import annotations

import os
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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

        tool_calls_payload = None

        def _normalize_tool_calls(tc_list: list) -> list[dict]:
            mapped: List[dict] = []
            for call in tc_list:
                if not isinstance(call, dict):
                    continue
                try:
                    call_id = call.get("id") or "toolcall_0"
                    if isinstance(call.get("function"), dict):
                        fn = call["function"]
                        fn_name = fn.get("name") or call.get("name") or "tool"
                        fn_args = fn.get("arguments")
                    else:
                        fn_name = call.get("name") or "tool"
                        fn_args = call.get("arguments") if call.get("arguments") is not None else call.get("args")
                    if not isinstance(fn_args, str):
                        import json as _json

                        fn_args = _json.dumps(fn_args or {}, ensure_ascii=False)
                    mapped.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": fn_args},
                        }
                    )
                except Exception:
                    continue
            return mapped

        ak = getattr(msg, "additional_kwargs", None)
        if isinstance(ak, dict):
            tc = ak.get("tool_calls")
            if isinstance(tc, list) and tc:
                mapped = _normalize_tool_calls(tc)
                if mapped:
                    tool_calls_payload = mapped

        if tool_calls_payload is None:
            raw_attr_tc = getattr(msg, "tool_calls", None)
            if isinstance(raw_attr_tc, list) and raw_attr_tc:
                mapped = _normalize_tool_calls(raw_attr_tc)
                if mapped:
                    tool_calls_payload = mapped

        # Extract reasoning/thinking parts into reasoning_content
        reasoning_content = None
        if isinstance(msg.content, list):
            collected = [
                it.get("thinking", "") for it in msg.content if isinstance(it, dict) and it.get("type") == "thinking"
            ]
            if collected:
                reasoning_content = "\n\n".join([s for s in collected if s]) or None

        ep_msg = Message(
            role="assistant", content=content, tool_calls=tool_calls_payload, reasoning_content=reasoning_content
        )
        _dbg_print(
            "[EP-Ser] -> EP Message:",
            {
                "role": ep_msg.role,
                "content_len": len(ep_msg.content or ""),
                "tool_calls": len(ep_msg.tool_calls or []) if isinstance(ep_msg.tool_calls, list) else 0,
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
