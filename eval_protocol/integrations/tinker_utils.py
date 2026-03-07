from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from eval_protocol.models import Message as EvalProtocolMessage

try:
    import tinker
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.model_info import get_recommended_renderer_name

    try:
        from tinker_cookbook.image_processing_utils import get_image_processor
    except ImportError:  # pragma: no cover - older Tinker fallback
        get_image_processor = None  # type: ignore[assignment]

    TINKER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    tinker = None  # type: ignore[assignment]
    renderers = None  # type: ignore[assignment]
    tokenizer_utils = None  # type: ignore[assignment]
    get_recommended_renderer_name = None  # type: ignore[assignment]
    get_image_processor = None  # type: ignore[assignment]
    TINKER_AVAILABLE = False


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def resolve_renderer_name(model_name: str, renderer_name: str = "") -> str:
    if renderer_name:
        return renderer_name
    if "moonshotai/kimi-k2.5" in model_name.lower():
        return "kimi_k25"
    if get_recommended_renderer_name is None:
        raise ImportError("tinker-cookbook is required to resolve Tinker renderer names")
    return get_recommended_renderer_name(model_name)


def _renderer_uses_images(renderer_name: str) -> bool:
    return any(
        marker in renderer_name
        for marker in (
            "_vl",
            "qwen3_5",
            "kimi_k25",
        )
    )


def build_tinker_renderer(
    *,
    model_name: str,
    renderer_name: str = "",
    tokenizer: Any | None = None,
) -> tuple[Any, Any]:
    if not TINKER_AVAILABLE:
        raise ImportError("tinker-cookbook is required to use the Tinker integrations")

    resolved_name = resolve_renderer_name(model_name, renderer_name)
    if tokenizer is None:
        tokenizer = tokenizer_utils.get_tokenizer(model_name)

    kwargs: dict[str, Any] = {}
    if get_image_processor is not None and _renderer_uses_images(resolved_name):
        kwargs["image_processor"] = get_image_processor(model_name)

    renderer = renderers.get_renderer(resolved_name, tokenizer=tokenizer, **kwargs)
    return tokenizer, renderer


def _normalize_image_part(part: Mapping[str, Any]) -> dict[str, Any]:
    image_value = _field(part, "image")
    if image_value is not None:
        return {"type": "image", "image": image_value}

    image_url = _field(part, "image_url")
    if isinstance(image_url, str):
        return {"type": "image", "image": image_url}
    if isinstance(image_url, Mapping) and isinstance(image_url.get("url"), str):
        return {"type": "image", "image": image_url["url"]}
    raise TypeError(f"Unsupported image content part: {part!r}")


def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        part_type = _field(content, "type")
        if part_type in {"image", "image_url"}:
            return [_normalize_image_part(content)]
        if part_type == "thinking" and isinstance(_field(content, "thinking"), str):
            return [{"type": "thinking", "thinking": _field(content, "thinking")}]
        if isinstance(_field(content, "text"), str):
            return _field(content, "text")
        raise TypeError(f"Unsupported message content mapping: {content!r}")
    if isinstance(content, Sequence):
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, Mapping):
                raise TypeError(f"Unsupported message content part: {part!r}")
            part_type = _field(part, "type")
            if part_type == "text" and isinstance(_field(part, "text"), str):
                parts.append({"type": "text", "text": _field(part, "text")})
                continue
            if part_type in {"image", "image_url"}:
                parts.append(_normalize_image_part(part))
                continue
            if part_type == "thinking" and isinstance(_field(part, "thinking"), str):
                parts.append({"type": "thinking", "thinking": _field(part, "thinking")})
                continue
            raise TypeError(f"Unsupported message content part: {part!r}")
        if parts and all(part["type"] == "text" for part in parts):
            return "".join(str(part["text"]) for part in parts)
        return parts
    raise TypeError(f"Unsupported message content type: {type(content)!r}")


def _ensure_content_parts(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _normalize_tool_calls(tool_calls: Any) -> list[Any]:
    if renderers is None:
        raise ImportError("tinker-cookbook is required to normalize tool calls")

    normalized = []
    for tool_call in tool_calls or []:
        name = None
        arguments = None
        tool_call_id = _field(tool_call, "id")

        if isinstance(_field(tool_call, "name"), str):
            name = _field(tool_call, "name")
            arguments = _field(tool_call, "args")
        else:
            function = _field(tool_call, "function")
            name = _field(function, "name")
            arguments = _field(function, "arguments")

        if not isinstance(name, str):
            raise ValueError(f"Unsupported tool call shape: {tool_call!r}")
        if isinstance(arguments, str):
            parsed_arguments = json.loads(arguments) if arguments else {}
        elif isinstance(arguments, Mapping):
            parsed_arguments = dict(arguments)
        else:
            raise TypeError(f"Unsupported tool call arguments type: {type(arguments)!r}")

        normalized.append(
            renderers.ToolCall(
                function=renderers.ToolCall.FunctionBody(
                    name=name,
                    arguments=json.dumps(parsed_arguments),
                ),
                id=tool_call_id,
            )
        )
    return normalized


def normalize_eval_protocol_messages(messages: Sequence[EvalProtocolMessage | Mapping[str, Any]]) -> list[Any]:
    if not TINKER_AVAILABLE:
        raise ImportError("tinker-cookbook is required to normalize Tinker messages")

    normalized = []
    for raw_message in messages:
        message = (
            raw_message.model_dump(exclude_none=True)
            if hasattr(raw_message, "model_dump")
            else dict(raw_message)
        )

        role = _field(message, "role")
        if not isinstance(role, str):
            raise ValueError(f"Message is missing a string role: {message!r}")

        normalized_message = {
            "role": role,
            "content": _normalize_content(_field(message, "content")),
        }

        tool_calls = _field(message, "tool_calls")
        if tool_calls is not None:
            normalized_message["tool_calls"] = _normalize_tool_calls(tool_calls)

        reasoning_content = _field(message, "reasoning_content")
        if reasoning_content is not None:
            if not isinstance(reasoning_content, str):
                raise TypeError(
                    f"Unsupported reasoning_content type: {type(reasoning_content)!r}"
                )
            normalized_message["content"] = [
                {"type": "thinking", "thinking": reasoning_content},
                *_ensure_content_parts(normalized_message["content"]),
            ]

        tool_call_id = _field(message, "tool_call_id")
        if tool_call_id is not None:
            normalized_message["tool_call_id"] = str(tool_call_id)

        name = _field(message, "name")
        if name is not None:
            normalized_message["name"] = str(name)

        weight = _field(message, "weight")
        if weight is not None:
            normalized_message["trainable"] = bool(weight)

        normalized.append(normalized_message)

    return normalized


def _convert_content_for_eval_protocol(content: Any) -> tuple[str | list[dict[str, Any]], str | None]:
    reasoning_parts: list[str] = []
    if isinstance(content, str):
        return content, None

    content_parts: list[dict[str, Any]] = []
    for part in content or []:
        part_type = _field(part, "type")
        if part_type == "thinking":
            thinking = _field(part, "thinking")
            if isinstance(thinking, str):
                reasoning_parts.append(thinking)
            continue
        if part_type == "text":
            content_parts.append({"type": "text", "text": _field(part, "text") or ""})
            continue
        if part_type == "image":
            image_value = _field(part, "image")
            if not isinstance(image_value, str):
                raise TypeError(
                    "Eval Protocol message rendering only supports string image references for Tinker outputs."
                )
            content_parts.append(
                {"type": "image_url", "image_url": {"url": image_value}}
            )
            continue
        raise TypeError(f"Unsupported Tinker content part: {part!r}")

    if not content_parts:
        return "", "".join(reasoning_parts) or None
    if all(_field(part, "type") == "text" for part in content_parts):
        return "".join(str(_field(part, "text") or "") for part in content_parts), "".join(reasoning_parts) or None
    return content_parts, "".join(reasoning_parts) or None


def tinker_message_to_eval_protocol_message(message: Mapping[str, Any] | Any) -> EvalProtocolMessage:
    content, reasoning_content = _convert_content_for_eval_protocol(_field(message, "content", ""))
    payload: dict[str, Any] = {
        "role": _field(message, "role"),
        "content": content,
    }
    if reasoning_content:
        payload["reasoning_content"] = reasoning_content

    tool_calls = _field(message, "tool_calls")
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": _field(tool_call, "id"),
                "type": "function",
                "function": {
                    "name": _field(_field(tool_call, "function"), "name"),
                    "arguments": _field(_field(tool_call, "function"), "arguments"),
                },
            }
            for tool_call in tool_calls
        ]

    tool_call_id = _field(message, "tool_call_id")
    if tool_call_id is not None:
        payload["tool_call_id"] = str(tool_call_id)

    name = _field(message, "name")
    if name is not None:
        payload["name"] = str(name)

    return EvalProtocolMessage.model_validate(payload)
