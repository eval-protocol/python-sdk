"""Generic local-tokenized Fireworks /v1/completions client for tool-call rollouts.

This client handles:
  - Local tokenization via HuggingFace ``transformers``
  - Prompt construction via ``apply_chat_template``
  - Calling the ``/v1/completions`` endpoint with token-in / token-out
  - Logprob extraction
  - Retries for transient errors

Tool-call parsing is **not** built in.  Pass a ``tool_call_parser`` callback
to have the client include structured tool-call data in its response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from fireworks import AsyncFireworks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic types — usable by any tool-call domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedToolCall:
    tool_call_id: str
    name: str
    arguments: Dict[str, Any]


def to_openai_tool_calls(tool_call: ParsedToolCall) -> List[Dict[str, Any]]:
    """Convert a ``ParsedToolCall`` into OpenAI-compatible ``tool_calls`` payload."""
    return [
        {
            "id": tool_call.tool_call_id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments, separators=(",", ":")),
            },
        }
    ]


ToolCallParserFn = Callable[
    [str, List[int], Optional[List[Dict[str, Any]]]],
    Dict[str, Any],
]
"""Signature: ``(completion_text, completion_token_ids, tools) -> result_dict``.

The returned dict should contain:
  - ``parsed_tool_call``: a :class:`ParsedToolCall`
  - ``assistant_content``: ``str`` (text content outside the tool call)
  - ``parser``: ``str`` (name of the parsing strategy that succeeded)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_token_id_sequence(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = values.get("input_ids", values.get("ids", []))
    if values is None:
        return []
    if hasattr(values, "tolist") and not isinstance(values, list):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    return [int(x) for x in list(values)]


def _coerce_message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)


def _sanitize_messages_for_template(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        sanitized_msg: Dict[str, Any] = {
            "role": role,
            "content": _coerce_message_content_to_text(msg.get("content")),
        }
        if msg.get("tool_calls") is not None:
            sanitized_msg["tool_calls"] = msg.get("tool_calls")
        if msg.get("tool_call_id") is not None:
            sanitized_msg["tool_call_id"] = msg.get("tool_call_id")
        if msg.get("name") is not None:
            sanitized_msg["name"] = msg.get("name")
        sanitized.append(sanitized_msg)
    return sanitized


def _build_fallback_prompt_text(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
    chunks: List[str] = []
    if tools:
        chunks.append("TOOLS:")
        for tool in tools:
            function = tool.get("function", {})
            chunks.append(
                json.dumps(
                    {
                        "name": function.get("name"),
                        "description": function.get("description"),
                        "parameters": function.get("parameters"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        chunks.append("")
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = _coerce_message_content_to_text(msg.get("content"))
        chunks.append(f"{role}: {content}")
        if msg.get("tool_calls"):
            chunks.append(f"{role}_TOOL_CALLS: {json.dumps(msg['tool_calls'], ensure_ascii=False)}")
    chunks.append("ASSISTANT:")
    return "\n".join(chunks)


def strip_chat_special_tokens(text: str) -> str:
    """Remove common chat-template special tokens from text."""
    cleaned = str(text or "")
    for marker in ("<|im_end|>", "<|im_start|>"):
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FireworksV1CompletionsClient:
    """Adapter that performs local tokenization before ``/v1/completions`` calls.

    Parameters
    ----------
    tool_call_parser:
        Optional callback that extracts structured tool-call information from
        the raw completion text.  When *None*, the response ``choices[0].message``
        will contain the raw text with no ``tool_calls``.
    default_tools:
        Fallback tools list used when none is passed to individual calls.
    """

    def __init__(
        self,
        *,
        model_id: str,
        tokenizer_name_or_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        request_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = True,
        enable_thinking: Optional[bool] = None,
        tool_call_parser: Optional[ToolCallParserFn] = None,
        default_tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model_id = model_id
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_params = dict(request_params or {})
        self.logprobs = logprobs
        self.enable_thinking = enable_thinking
        self.tool_call_parser = tool_call_parser
        self.default_tools = default_tools or []
        self._tokenizer = None
        self._assistant_prefix_token_ids: Optional[List[int]] = None
        self._client = AsyncFireworks(api_key=api_key, base_url=base_url)

    async def close(self) -> None:
        await self._client.close()

    # -- Tokenizer ----------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for local tokenizer mode. "
                    "Install a build with transformers support (for example, eval-protocol[dev])."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=True)
        return self._tokenizer

    def _get_assistant_prefix_token_ids(self) -> List[int]:
        if self._assistant_prefix_token_ids is None:
            tokenizer = self._get_tokenizer()
            self._assistant_prefix_token_ids = _normalize_token_id_sequence(
                tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            )
        return list(self._assistant_prefix_token_ids)

    def _thinking_kwargs(self) -> Dict[str, Any]:
        if self.enable_thinking is not None:
            return {"enable_thinking": self.enable_thinking}
        return {}

    # -- Prompt building ----------------------------------------------------

    def _build_prompt_token_ids(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> List[int]:
        tokenizer = self._get_tokenizer()
        sanitized_messages = _sanitize_messages_for_template(messages=messages)
        thinking_kw = self._thinking_kwargs()
        token_ids: Any
        try:
            token_ids = tokenizer.apply_chat_template(
                sanitized_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                **thinking_kw,
            )
        except Exception as exc:
            if tools:
                logger.debug("Tokenizer chat template with tools failed, retrying without tools: %s", exc)
                try:
                    token_ids = tokenizer.apply_chat_template(
                        sanitized_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        **thinking_kw,
                    )
                except Exception as exc_no_tools:
                    logger.debug("Tokenizer chat template failed, using fallback text prompt: %s", exc_no_tools)
                    fallback_prompt = _build_fallback_prompt_text(messages=sanitized_messages, tools=tools)
                    token_ids = tokenizer.encode(fallback_prompt, add_special_tokens=False)
            else:
                logger.debug("Tokenizer chat template failed, using fallback text prompt: %s", exc)
                fallback_prompt = _build_fallback_prompt_text(messages=sanitized_messages, tools=tools)
                token_ids = tokenizer.encode(fallback_prompt, add_special_tokens=False)

        return _normalize_token_id_sequence(token_ids)

    def build_prompt_token_ids(self, *, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> List[int]:
        """Public wrapper used by rollout processors to initialize token history."""
        return self._build_prompt_token_ids(messages=messages, tools=tools)

    def build_tool_response_suffix_token_ids(self, *, tool_message: Dict[str, Any]) -> List[int]:
        """Build token ids for appending a tool response turn and next assistant prefix."""
        tokenizer = self._get_tokenizer()
        sanitized_messages = _sanitize_messages_for_template(messages=[tool_message])
        thinking_kw = self._thinking_kwargs()
        token_ids: Any
        try:
            token_ids = tokenizer.apply_chat_template(
                sanitized_messages,
                tokenize=True,
                add_generation_prompt=True,
                **thinking_kw,
            )
        except Exception as exc:
            logger.debug("Tokenizer tool suffix template failed, using fallback text prompt: %s", exc)
            fallback_prompt = _build_fallback_prompt_text(messages=sanitized_messages, tools=None)
            token_ids = tokenizer.encode(fallback_prompt, add_special_tokens=False)
        return _normalize_token_id_sequence(token_ids)

    def build_assistant_turn_token_ids(self, *, assistant_message: Dict[str, Any]) -> List[int]:
        """Build canonical assistant tool-call turn tokens (without generation prompt)."""
        tokenizer = self._get_tokenizer()
        sanitized_messages = _sanitize_messages_for_template(messages=[assistant_message])
        token_ids: Any
        thinking_kw = self._thinking_kwargs()
        try:
            token_ids = tokenizer.apply_chat_template(
                sanitized_messages,
                tokenize=True,
                add_generation_prompt=False,
                **thinking_kw,
            )
        except Exception as exc:
            logger.debug("Tokenizer assistant turn template failed, using fallback text prompt: %s", exc)
            fallback_prompt = _build_fallback_prompt_text(messages=sanitized_messages, tools=None)
            token_ids = tokenizer.encode(fallback_prompt, add_special_tokens=False)
        normalized = _normalize_token_id_sequence(token_ids)
        assistant_prefix = self._get_assistant_prefix_token_ids()
        if assistant_prefix and normalized[: len(assistant_prefix)] == assistant_prefix:
            return normalized[len(assistant_prefix) :]
        return normalized

    def encode_special_suffix(self) -> List[int]:
        """Return token IDs for ``<|im_end|>\\n`` — the end-of-turn marker."""
        tokenizer = self._get_tokenizer()
        return _normalize_token_id_sequence(
            tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        )

    def decode_token_ids(self, *, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        tokenizer = self._get_tokenizer()
        try:
            return str(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(tokenizer.decode(token_ids))

    # -- Completion ---------------------------------------------------------

    async def create_completion_from_prompt_ids(
        self,
        *,
        prompt_token_ids: List[int],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call ``/v1/completions`` and return a structured result dict.

        If ``tool_call_parser`` was provided at construction time, the result
        will include ``choices[0].message.tool_calls``.  Otherwise the message
        will contain only the raw ``content``.
        """
        active_tools = tools if tools is not None else (self.default_tools or None)
        normalized_prompt_token_ids = [int(x) for x in list(prompt_token_ids)]
        request_payload = {
            **self.request_params,
            "model": self.model_id,
            "prompt": normalized_prompt_token_ids,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True if self.logprobs else None,
        }
        if not self.logprobs:
            request_payload.pop("logprobs", None)

        max_retries = 40
        base_delay = 10.0
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.completions.create(**request_payload)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                err_str = str(exc)
                is_transient = (
                    status in (425, 429, 502, 503, 504)
                    or "model_not_ready" in err_str
                    or "hot loading" in err_str
                    or "Model not found" in err_str
                    or "DEPLOYMENT_SCALING_UP" in err_str
                )
                if not is_transient or attempt >= max_retries:
                    raise
                delay = min(base_delay * (2 ** attempt), 60.0)
                logger.info(
                    "Retryable error (attempt %d/%d, status=%s), retrying in %.1fs: %s",
                    attempt + 1, max_retries, status, delay, err_str[:200],
                )
                await asyncio.sleep(delay)

        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        choices = response_dict.get("choices") or []
        if not choices:
            raise ValueError("Fireworks /v1/completions response did not include choices")

        choice = choices[0]
        finish_reason = str(choice.get("finish_reason") or "unknown")

        raw_output = choice.get("raw_output") if isinstance(choice.get("raw_output"), dict) else {}
        completion_token_ids = _normalize_token_id_sequence(
            choice.get("token_ids") or raw_output.get("completion_token_ids") or []
        )
        choice_prompt_token_ids = _normalize_token_id_sequence(
            choice.get("prompt_token_ids") or raw_output.get("prompt_token_ids") or normalized_prompt_token_ids
        )

        completion_text = self.decode_token_ids(token_ids=completion_token_ids)
        if not completion_text:
            completion_text = str(choice.get("text") or "")
        if not completion_token_ids and completion_text:
            tokenizer = self._get_tokenizer()
            completion_token_ids = list(tokenizer.encode(completion_text, add_special_tokens=False))

        # -- Extract logprobs -----------------------------------------------
        completion_logprobs: List[float] = []
        choice_logprobs = choice.get("logprobs")
        if isinstance(choice_logprobs, dict):
            token_logprobs = choice_logprobs.get("token_logprobs") or []
            if token_logprobs:
                completion_logprobs = [float(lp) if lp is not None else 0.0 for lp in token_logprobs]
            else:
                content_logprobs = choice_logprobs.get("content") or []
                completion_logprobs = [
                    float(entry.get("logprob", 0.0)) if isinstance(entry, dict) else 0.0
                    for entry in content_logprobs
                ]
        elif isinstance(choice_logprobs, list):
            completion_logprobs = [float(lp) if lp is not None else 0.0 for lp in choice_logprobs]

        # -- Build message via parser or raw --------------------------------
        if self.tool_call_parser is not None:
            parsed_output = self.tool_call_parser(completion_text, completion_token_ids, active_tools)
            parsed_tool_call: Optional[ParsedToolCall] = parsed_output.get("parsed_tool_call")
            assistant_content = str(parsed_output.get("assistant_content", "") or "")
            parser_name = str(parsed_output.get("parser", "external"))
            message_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if parsed_tool_call is not None:
                message_payload["tool_calls"] = to_openai_tool_calls(parsed_tool_call)
        else:
            assistant_content = strip_chat_special_tokens(completion_text)
            parser_name = "none"
            message_payload = {"role": "assistant", "content": assistant_content}

        usage_obj = response_dict.get("usage") or {}
        usage_payload = {
            "prompt_tokens": int(usage_obj.get("prompt_tokens", len(choice_prompt_token_ids))),
            "completion_tokens": int(usage_obj.get("completion_tokens", len(completion_token_ids))),
            "total_tokens": int(
                usage_obj.get("total_tokens", len(choice_prompt_token_ids) + len(completion_token_ids))
            ),
        }

        result: Dict[str, Any] = {
            "choices": [
                {
                    "message": message_payload,
                    "finish_reason": finish_reason,
                    "raw_output": {**dict(raw_output or {}), "tool_call_parser": parser_name},
                }
            ],
            "usage": usage_payload,
            "prompt_ids": list(choice_prompt_token_ids),
            "completion_ids": list(completion_token_ids),
            "finish_reason": finish_reason,
            "raw_output": {**dict(raw_output or {}), "tool_call_parser": parser_name},
        }
        if completion_logprobs:
            result["completion_logprobs"] = completion_logprobs
        return result

    async def create_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """High-level helper: tokenize *messages* then call ``create_completion_from_prompt_ids``."""
        active_tools = tools if tools is not None else (self.default_tools or None)
        prompt_token_ids = self.build_prompt_token_ids(messages=messages, tools=active_tools)
        return await self.create_completion_from_prompt_ids(
            prompt_token_ids=prompt_token_ids,
            tools=active_tools,
        )
