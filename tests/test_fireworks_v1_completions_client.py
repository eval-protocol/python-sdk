import asyncio
from typing import Any, Dict, List, Optional

import pytest

from eval_protocol.integrations.fireworks_v1_completions_client import (
    FireworksV1CompletionsClient,
    ParsedToolCall,
    to_openai_tool_calls,
    strip_chat_special_tokens,
)


def test_parsed_tool_call_to_openai_format():
    tc = ParsedToolCall(tool_call_id="call_1", name="lake_move", arguments={"action": "RIGHT"})
    payload = to_openai_tool_calls(tc)
    assert len(payload) == 1
    assert payload[0]["function"]["name"] == "lake_move"
    assert '"action":"RIGHT"' in payload[0]["function"]["arguments"]


def test_strip_chat_special_tokens():
    assert strip_chat_special_tokens("<|im_start|>assistant\nhello<|im_end|>") == "assistant\nhello"
    assert strip_chat_special_tokens("") == ""
    assert strip_chat_special_tokens(None) == ""


def test_tool_call_parser_is_invoked():
    """When a tool_call_parser is provided, create_completion_from_prompt_ids uses it."""

    def fake_parser(
        text: str, ids: List[int], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        return {
            "parsed_tool_call": ParsedToolCall(
                tool_call_id="call_0", name="test_tool", arguments={"x": 1}
            ),
            "assistant_content": "thought",
            "parser": "fake",
        }

    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        tool_call_parser=fake_parser,
    )

    result = fake_parser("some text", [1, 2], None)
    assert result["parsed_tool_call"].name == "test_tool"
    assert result["assistant_content"] == "thought"
    asyncio.run(client.close())


def test_no_parser_returns_raw_content():
    """When no tool_call_parser is provided, message contains raw content."""
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    assert client.tool_call_parser is None
    asyncio.run(client.close())


def test_default_tools_not_used_when_tools_is_empty_list():
    """Passing tools=[] should not fall back to default_tools."""
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        default_tools=[{"type": "function", "function": {"name": "my_tool"}}],
    )
    assert client.default_tools == [{"type": "function", "function": {"name": "my_tool"}}]
    asyncio.run(client.close())


def test_build_prompt_token_ids_retries_without_tools(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )

    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("tools unsupported")
            return [11, 22, 33]

        def encode(self, text, add_special_tokens=False):
            return [99]

    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(client, "_get_tokenizer", lambda: fake_tokenizer)
    token_ids = client._build_prompt_token_ids(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lake_move"}}],
    )
    assert token_ids == [11, 22, 33]
    assert len(fake_tokenizer.calls) == 2
    asyncio.run(client.close())


def test_build_prompt_token_ids_handles_dict_input_ids(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [[101, 102, 103]]}

        def encode(self, text, add_special_tokens=False):
            return [99]

    monkeypatch.setattr(client, "_get_tokenizer", lambda: FakeTokenizer())
    token_ids = client._build_prompt_token_ids(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
    )
    assert token_ids == [101, 102, 103]
    asyncio.run(client.close())


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return self._payload


def _install_fake_completion(client, monkeypatch, payload):
    captured: Dict[str, Any] = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(payload)

    class _FakeCompletions:
        create = staticmethod(fake_create)

    class _FakeClient:
        completions = _FakeCompletions()

        async def close(self):
            return None

    monkeypatch.setattr(client, "_client", _FakeClient())
    return captured


def test_reads_ids_and_sampling_logprobs_from_content(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    monkeypatch.setattr(client, "decode_token_ids", lambda token_ids: "text")

    def _fail_tokenizer():
        raise AssertionError("tokenizer must not be used to re-encode completion text")

    monkeypatch.setattr(client, "_get_tokenizer", lambda: _fail_tokenizer())
    captured = _install_fake_completion(
        client,
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token_id": 271, "sampling_logprob": -0.05483185, "logprob": -0.0548313},
                            {"token_id": 248068, "sampling_logprob": -0.0014, "logprob": -0.0014},
                            {"token_id": 26108, "sampling_logprob": -1.0, "logprob": -0.99},
                        ]
                    },
                }
            ],
        },
    )
    result = asyncio.run(client.create_completion_from_prompt_ids(prompt_token_ids=[1, 2]))
    assert "return_token_ids" not in captured
    assert result["completion_ids"] == [271, 248068, 26108]
    assert result["completion_logprobs"] == [-0.05483185, -0.0014, -1.0]
    assert len(result["completion_ids"]) == len(result["completion_logprobs"])
    asyncio.run(client.close())


def test_raises_when_content_entry_missing_sampling_logprob(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    monkeypatch.setattr(client, "decode_token_ids", lambda token_ids: "text")
    _install_fake_completion(
        client,
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token_id": 1, "sampling_logprob": -0.1},
                            {"token_id": 2, "logprob": -0.2},
                        ]
                    },
                }
            ],
        },
    )
    with pytest.raises(RuntimeError, match="missing .*sampling_logprob"):
        asyncio.run(client.create_completion_from_prompt_ids(prompt_token_ids=[1]))
    asyncio.run(client.close())


def test_raises_when_no_content_logprobs(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    _install_fake_completion(
        client,
        monkeypatch,
        {"choices": [{"text": "hello world", "finish_reason": "stop"}]},
    )
    with pytest.raises(RuntimeError, match="no content\\[\\] logprobs entries"):
        asyncio.run(client.create_completion_from_prompt_ids(prompt_token_ids=[1, 2]))
    asyncio.run(client.close())


def test_ignores_legacy_token_logprobs_shape(monkeypatch):
    """The legacy token_logprobs shape has no content[]; the client must not use it."""
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    _install_fake_completion(
        client,
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "token_ids": [1, 2, 3],
                    "logprobs": {
                        "token_ids": [1, 2, 3],
                        "token_logprobs": [-0.1, -0.2, -0.3],
                    },
                }
            ],
        },
    )
    with pytest.raises(RuntimeError, match="no content\\[\\] logprobs entries"):
        asyncio.run(client.create_completion_from_prompt_ids(prompt_token_ids=[1]))
    asyncio.run(client.close())


def test_raises_when_content_entry_missing_token_id(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="test-model",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    monkeypatch.setattr(client, "decode_token_ids", lambda token_ids: "text")
    _install_fake_completion(
        client,
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token_id": 1, "sampling_logprob": -0.1},
                            {"sampling_logprob": -0.2},
                        ]
                    },
                }
            ],
        },
    )
    with pytest.raises(RuntimeError, match="missing token_id"):
        asyncio.run(client.create_completion_from_prompt_ids(prompt_token_ids=[1]))
    asyncio.run(client.close())


def test_thinking_kwargs_respects_enable_thinking():
    client_none = FireworksV1CompletionsClient(
        model_id="test", tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    assert client_none._thinking_kwargs() == {}

    client_false = FireworksV1CompletionsClient(
        model_id="test", tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        enable_thinking=False,
    )
    assert client_false._thinking_kwargs() == {"enable_thinking": False}

    client_true = FireworksV1CompletionsClient(
        model_id="test", tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        enable_thinking=True,
    )
    assert client_true._thinking_kwargs() == {"enable_thinking": True}
    asyncio.run(client_none.close())
    asyncio.run(client_false.close())
    asyncio.run(client_true.close())
